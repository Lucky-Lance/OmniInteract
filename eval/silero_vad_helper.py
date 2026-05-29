"""Streaming silero-vad helper.

Wraps the ``silero-vad`` package's ``VADIterator`` so that the evaluation
driver can push 16 kHz mono int16 PCM samples in arbitrary-size chunks and
consume ``(speech_started_sec, speech_stopped_sec)`` events.

The iterator internally requires fixed 512-sample windows at 16 kHz, so this
wrapper handles partial-frame buffering and converts the int16 PCM into the
``torch.float32`` tensor that the model expects.

Usage:

    vad = StreamingVAD(threshold=0.5, sampling_rate=16000,
                       min_silence_duration_ms=500, speech_pad_ms=100)
    for evt in vad.feed(int16_pcm_bytes):
        # evt = {'kind': 'start'|'end', 'sample': int, 'time': float}
        ...
    # At end of stream:
    for evt in vad.flush():
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import torch


_FRAME_SAMPLES_AT_16K = 512


@dataclass
class VADEvent:
    """A single speech-boundary event emitted by the streaming VAD."""

    kind: str  # 'start' or 'end'
    sample: int  # absolute sample index in the input stream
    time: float  # absolute seconds in the input stream


class StreamingVAD:
    """Thin wrapper around ``silero_vad.VADIterator`` for byte-stream feeding."""

    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
    ) -> None:
        if sampling_rate != 16000:
            raise ValueError(
                f"StreamingVAD only supports 16000 Hz input, got {sampling_rate}"
            )
        # Imported lazily so that ``import silero_vad_helper`` doesn't pay the
        # torchaudio/torch JIT-load cost when the module is just imported for
        # type information.
        from silero_vad import load_silero_vad, VADIterator

        self._sampling_rate = sampling_rate
        self._frame_samples = _FRAME_SAMPLES_AT_16K
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        # int16 PCM bytes left over from the previous feed() call (less than
        # one window).
        self._byte_buf = bytearray()
        # Absolute sample index processed so far (used to keep our own clock
        # because the silero iterator only reports per-frame offsets).
        self._samples_seen = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state (call between independent streams)."""
        self._iterator.reset_states()
        self._byte_buf.clear()
        self._samples_seen = 0

    # ------------------------------------------------------------------
    def feed(self, pcm_bytes: bytes) -> Iterator[VADEvent]:
        """Push int16 mono PCM bytes; yield boundary events as they fire."""
        if not pcm_bytes:
            return
        self._byte_buf.extend(pcm_bytes)
        frame_bytes = self._frame_samples * 2  # int16
        while len(self._byte_buf) >= frame_bytes:
            frame_int16 = np.frombuffer(
                bytes(self._byte_buf[:frame_bytes]), dtype=np.int16
            )
            del self._byte_buf[:frame_bytes]
            tensor = torch.from_numpy(
                frame_int16.astype(np.float32) / 32768.0
            )
            speech_dict = self._iterator(tensor, return_seconds=False)
            base_sample = self._samples_seen
            self._samples_seen += self._frame_samples
            if not speech_dict:
                continue
            # silero returns either {'start': sample_idx} or {'end': sample_idx}
            for key, sample_idx in speech_dict.items():
                evt_kind = "start" if key == "start" else "end"
                # ``sample_idx`` is the absolute sample count from the
                # iterator's perspective (it counts samples internally), so
                # we can use it directly.  Fall back to our own counter if
                # the value looks bogus.
                abs_sample = int(sample_idx)
                if abs_sample <= 0:
                    abs_sample = base_sample
                yield VADEvent(
                    kind=evt_kind,
                    sample=abs_sample,
                    time=abs_sample / self._sampling_rate,
                )

    # ------------------------------------------------------------------
    def flush(self) -> Iterator[VADEvent]:
        """Return any final speech-end event the iterator wants to emit.

        Internally pads the leftover bytes with zeros to a full window so the
        iterator gets a chance to close any in-flight speech segment.
        """
        if self._byte_buf:
            pad = self._frame_samples * 2 - len(self._byte_buf)
            if pad > 0:
                self._byte_buf.extend(b"\x00" * pad)
            yield from self.feed(b"")
            # Final partial window: feed an explicit silence frame.
        # Force a final silence window so any in-flight speech terminates.
        silence = np.zeros(self._frame_samples, dtype=np.int16).tobytes()
        yield from self.feed(silence)


# ------------------------------------------------------------------
# Higher-level convenience: cut speech segments out of a streaming source.
# ------------------------------------------------------------------
class SpeechSegmenter:
    """Buffers PCM and emits ``(start_sec, end_sec, pcm_bytes)`` segments.

    Holds enough trailing audio in a ring buffer so that when the VAD reports a
    speech start (which is back-dated by ``speech_pad_ms``) we can recover the
    PCM bytes from that earlier sample index.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        max_buffer_seconds: float = 60.0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.vad = StreamingVAD(
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        # Absolute sample index of the first byte still in self._buf.
        self._buf_start_sample = 0
        self._buf = bytearray()
        self._max_buf_bytes = int(max_buffer_seconds * sampling_rate) * 2
        self._active_start_sample: Optional[int] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.vad.reset()
        self._buf_start_sample = 0
        self._buf.clear()
        self._active_start_sample = None

    # ------------------------------------------------------------------
    def feed(self, pcm_bytes: bytes):
        """Append PCM, run VAD, yield (event_type, ...) tuples.

        Possible yields:
          - ('start', start_sec)
          - ('segment', start_sec, end_sec, segment_pcm_bytes)
        """
        if pcm_bytes:
            self._buf.extend(pcm_bytes)
        results: List[tuple] = []
        for evt in self.vad.feed(pcm_bytes):
            if evt.kind == "start":
                self._active_start_sample = evt.sample
                results.append(("start", evt.sample / self.sampling_rate))
            elif evt.kind == "end":
                if self._active_start_sample is None:
                    continue
                start = self._active_start_sample
                end = evt.sample
                self._active_start_sample = None
                seg = self._extract(start, end)
                if seg is not None:
                    results.append(
                        ("segment",
                         start / self.sampling_rate,
                         end / self.sampling_rate,
                         seg)
                    )
        # Trim the ring buffer so it doesn't grow unbounded.
        if self._active_start_sample is not None:
            keep_from = self._active_start_sample
        else:
            # Keep enough trailing audio to recover a back-dated start event.
            keep_from = self._buf_start_sample + max(
                0, len(self._buf) - self._max_buf_bytes
            ) // 2 * 2
        self._drop_before(keep_from)
        return results

    # ------------------------------------------------------------------
    def flush(self):
        results: List[tuple] = []
        for evt in self.vad.flush():
            if evt.kind == "end" and self._active_start_sample is not None:
                start = self._active_start_sample
                end = evt.sample
                self._active_start_sample = None
                seg = self._extract(start, end)
                if seg is not None:
                    results.append(
                        ("segment",
                         start / self.sampling_rate,
                         end / self.sampling_rate,
                         seg)
                    )
        return results

    # ------------------------------------------------------------------
    def _extract(self, start_sample: int, end_sample: int) -> Optional[bytes]:
        if end_sample <= start_sample:
            return None
        byte_start = (start_sample - self._buf_start_sample) * 2
        byte_end = (end_sample - self._buf_start_sample) * 2
        if byte_start < 0:
            # Audio for the start has already been dropped (segment would be
            # truncated); back off to whatever we still have buffered.
            byte_start = 0
        if byte_end > len(self._buf):
            byte_end = len(self._buf)
        if byte_end <= byte_start:
            return None
        return bytes(self._buf[byte_start:byte_end])

    # ------------------------------------------------------------------
    def _drop_before(self, sample_idx: int) -> None:
        if sample_idx <= self._buf_start_sample:
            return
        drop_bytes = (sample_idx - self._buf_start_sample) * 2
        if drop_bytes >= len(self._buf):
            self._buf.clear()
            self._buf_start_sample = sample_idx
        else:
            del self._buf[:drop_bytes]
            self._buf_start_sample = sample_idx


__all__ = ["StreamingVAD", "SpeechSegmenter", "VADEvent"]
