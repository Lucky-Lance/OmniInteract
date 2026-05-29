# Turn-based Chat FAQ

## What is Turn-based Chat mode?

Turn-based Chat is the classic question-and-answer conversation mode. You can send text, audio input, or video input, and the model will generate text and voice responses. Suitable for offline testing and prompt debugging.

## What input types are supported?

- **Text**: Type your questions directly
- **Audio**: Upload audio files as input
- **Video**: Upload video files as input
- Multimodal mixed input is supported

## About System Prompt

MiniCPM-o 4.5 introduces the concept of multimodal system prompts, allowing users to include text instructions, reference audio, etc. in the system prompt. The model will adjust its text style and voice prosody based on the text instructions and reference audio.

In the configuration card at the top, expand the **System Prompt** section to edit it. The System Prompt is sent to the model at the beginning of each conversation to define the role and behavior.

- About Reference Audio

In the System Prompt configuration section, upload a reference audio clip. The model will mimic the voice characteristics of the audio for its voice responses. Supports WAV / MP3 formats, recommended duration is 5-15 seconds.

## Mode Switching (Voice, Video Understanding, Text Chat)

You can switch between different presets via the System Prompt, or customize the system prompt directly.

## When should I enable voice response?

For spoken conversations, enable the Voice Response toggle. For written conversations, such as offline video analysis or markdown-formatted responses, disable the Voice Response toggle. When Voice Response is disabled, the model will generate text-only responses.

## Does enabling streaming affect quality?

Yes, it can. If you find the streaming quality unsatisfactory, try using non-streaming generation instead, which has a longer wait time.

## What should I do if the connection status shows Offline?

- Confirm the service has been started (`bash start_all.sh`)
- Check that the Gateway and Worker processes are running normally
- Look for WebSocket errors in the browser console
- Verify the access address and port are correct
