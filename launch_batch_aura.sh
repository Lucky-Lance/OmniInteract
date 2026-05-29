#!/bin/bash
# ========================================================================
# AURA-8b 全双工 Benchmark 批量推理启动脚本
#
# 启动 2 个完整 AURA 实例（各含 ASR + TTS + vLLM），并行处理 250 个视频。
# 每个实例占用 4 张 GPU：1 张 ASR，1 张 TTS，2 张 vLLM（TP=2）。
#
# 数据集 (OmniInteract data/ 布局):
#   data/1q1a       -> 150 视频 (video_json_map.json 配对 videos/ <-> annotations/)
#   data/1q1a_math  ->  60 视频 (同上)
#   data/1qna       ->  40 视频 (videos_bench/ 镜像 annotations/)
#   合计 250
#
# 用法:
#   bash launch_batch_aura.sh                    # 断点续跑（自动跳过已完成）
#   FORCE_RERUN=1 bash launch_batch_aura.sh      # 强制全部重跑
# ========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 配置 ── (路径用环境变量覆盖，默认值为占位符)
MODEL_PATH="${MODEL_PATH:-/path/to/AURA-8b}"
ASR_MODEL="${ASR_MODEL:-/path/to/Qwen3-ASR-1.7B}"
TTS_MODEL="${TTS_MODEL:-/path/to/Qwen3-TTS-12Hz-1.7B-Base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/aura}"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/data}"
SKIP_FLAG="--skip_existing"
if [ "${FORCE_RERUN:-0}" = "1" ]; then
    SKIP_FLAG=""
fi

LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

# ── GPU 分配 ──
# 实例 A: GPU 0 (ASR), GPU 1 (TTS), GPUs 2,3 (vLLM TP=2)
# 实例 B: GPU 4 (ASR), GPU 5 (TTS), GPUs 6,7 (vLLM TP=2)
# GPUs 8,9 空闲
INST_A_GPU_ASR=0
INST_A_GPU_TTS=1
INST_A_GPU_VLLM="2,3"
INST_A_ASR_PORT=8001
INST_A_TTS_PORT=8002
INST_A_VLLM_PORT=12346

INST_B_GPU_ASR=4
INST_B_GPU_TTS=5
INST_B_GPU_VLLM="6,7"
INST_B_ASR_PORT=8003
INST_B_TTS_PORT=8004
INST_B_VLLM_PORT=12347

TP_SIZE=2

ALL_PIDS=()

# ── 清理函数 ──
_do_cleanup() {
    echo ""
    echo "=========================================="
    echo "  Shutting down all services..."
    echo "=========================================="
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "  Stopped PID $pid"
        fi
    done
    sleep 2
    for pid in "${ALL_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "  All processes stopped."
}
trap '_do_cleanup; exit 130' SIGINT
trap '_do_cleanup; exit 143' SIGTERM
trap '_do_cleanup' EXIT

kill_port() {
    local port=$1
    local pids
    pids=$(ss -tlnp "sport = :$port" 2>/dev/null \
           | awk 'NR>1{match($0,/pid=([0-9]+)/,a); if(a[1]) print a[1]}' \
           | sort -u)
    if [ -n "$pids" ]; then
        echo "[cleanup] Port $port occupied by PID(s): $pids — killing..."
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 2
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# ── 清理残留端口 ──
echo "[init] Cleaning up leftover processes..."
for PORT in $INST_A_ASR_PORT $INST_A_TTS_PORT $INST_A_VLLM_PORT \
            $INST_B_ASR_PORT $INST_B_TTS_PORT $INST_B_VLLM_PORT; do
    kill_port $PORT
done

# ========================================================================
# 启动单个 AURA 实例的服务栈
# 参数: <实例名> <GPU_ASR> <GPU_TTS> <GPU_VLLM> <ASR_PORT> <TTS_PORT> <VLLM_PORT>
# ========================================================================
start_instance() {
    local NAME=$1
    local GPU_ASR=$2
    local GPU_TTS=$3
    local GPU_VLLM=$4
    local ASR_PORT=$5
    local TTS_PORT=$6
    local VLLM_PORT=$7

    echo ""
    echo "===== Starting instance $NAME ====="
    echo "  ASR GPU: $GPU_ASR (port $ASR_PORT)"
    echo "  TTS GPU: $GPU_TTS (port $TTS_PORT)"
    echo "  vLLM GPUs: $GPU_VLLM (port $VLLM_PORT, TP=$TP_SIZE)"

    # ── ASR ──
    echo "  [ASR] Starting (model: $ASR_MODEL)..."
    CUDA_VISIBLE_DEVICES=$GPU_ASR \
        python -u Qwen3_asr_serve.py \
            --host 0.0.0.0 --port $ASR_PORT \
            --model "$ASR_MODEL" \
            --gpu-memory-utilization 0.6 --no-forced-aligner \
        > "$LOG_DIR/${NAME}_asr.log" 2>&1 &
    ALL_PIDS+=($!)
    echo "    PID=$!, log: ${NAME}_asr.log"

    echo "  [ASR] Waiting for ready..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${ASR_PORT}/docs" > /dev/null 2>&1; then
            echo "    ASR ready."
            break
        fi
        if ! kill -0 "${ALL_PIDS[-1]}" 2>/dev/null; then
            echo "    ASR exited unexpectedly! Check $LOG_DIR/${NAME}_asr.log"
            return 1
        fi
        sleep 2
    done

    # ── TTS ──
    echo "  [TTS] Starting (model: $TTS_MODEL)..."
    CUDA_VISIBLE_DEVICES=$GPU_TTS \
        python -u tts_service.py \
            --port $TTS_PORT --gpu 0 \
            --model "$TTS_MODEL" \
            --language Chinese \
            --ref-audio "${TTS_REF_AUDIO:-$SCRIPT_DIR/shuhan.mp3}" \
            --ref-text "读书指通过阅读书籍获取知识、交流思想的行为，包含默读、朗读、精读及泛读等形式，是提升自我、了解世界的途径。该行为历史悠久，现代已延伸出数字化"听书"和视频讲书等多样化方式，其不仅是学习课程，更关乎思想文化评论。以下是关于"读书"的更多细节：定义与内涵：读书既可以是具体的阅读一本书，也泛指求学（如"上学读书"）。它是获取历史、文化、科学等知识的重要途径。方法与形式：传统方式包括精读、泛读、浏览；现代数字化读书（如网易云阅读、微信读书）让获取内容更加便捷。意义与目的：读书能提升身心修养、扩展视野、明确世界观与价值观，特别是一些经典著作和名人传记具有立志作用。文化载体：除了书籍本身，还有专门讨论书的学术刊物，如三联书店出版的《读书》杂志。常见现象：部分人在剧烈活动后读书容易感到困倦（一看书就困症结）。常用读书平台及榜单包括豆瓣读书、微信读书（推荐如《三体》、《活着》等）、国家智慧教育读书平台。" \
        > "$LOG_DIR/${NAME}_tts.log" 2>&1 &
    ALL_PIDS+=($!)
    echo "    PID=$!, log: ${NAME}_tts.log"

    echo "  [TTS] Waiting for ready..."
    for i in $(seq 1 180); do
        if curl -s "http://localhost:${TTS_PORT}/v1/tts/health" 2>/dev/null \
            | grep -q '"status":"ok"'; then
            echo "    TTS ready."
            break
        fi
        if ! kill -0 "${ALL_PIDS[-1]}" 2>/dev/null; then
            echo "    TTS exited unexpectedly! Check $LOG_DIR/${NAME}_tts.log"
            return 1
        fi
        sleep 2
    done

    # ── vLLM ──
    echo "  [vLLM] Starting..."
    CUDA_VISIBLE_DEVICES=$GPU_VLLM TP_SIZE=$TP_SIZE \
        python -u Qwen3_VL_online_streaming_v2_ContextManaged.py \
            --listen-port $VLLM_PORT \
            --model "$MODEL_PATH" \
            --tensor-parallel-size $TP_SIZE \
            --max-model-len 262144 \
            --max-seq-len 262144 \
            --gpu-memory-utilization 0.9 \
            --asr-url "http://localhost:${ASR_PORT}/asr" \
            --kv-offloading-size 10 \
            --disable-hybrid-kv-cache-manager \
            --block-size 16 \
            --prefix-caching-hash-algo xxhash \
            --mm-encoder-attn-backend FLASH_ATTN \
            --mm-encoder-tp-mode data \
            --max-num-batched-tokens 15360 \
            --temperature 0.5 \
            --max-tokens 128 \
            --enable-tts \
            --tts-service-url "http://localhost:${TTS_PORT}" \
            --tts-output-dir "$LOG_DIR/${NAME}_tts_results" \
            --cross-turn-penalty 1 \
            --cross-turn-lookback 10 \
            --cross-turn-ngram-sizes \
            --enable-pruning \
            --max-rounds 45 \
            --num-rounds-keep 30 \
            --max-context-qas 10 \
        > "$LOG_DIR/${NAME}_vllm.log" 2>&1 &
    ALL_PIDS+=($!)
    echo "    PID=$!, log: ${NAME}_vllm.log"

    echo "  [vLLM] Waiting for port $VLLM_PORT (may take several minutes)..."
    local READY=0
    for i in $(seq 1 600); do
        if ss -tln 2>/dev/null | grep -q ":${VLLM_PORT} "; then
            echo "    vLLM ready."
            READY=1
            break
        fi
        if ! kill -0 "${ALL_PIDS[-1]}" 2>/dev/null; then
            echo "    vLLM exited unexpectedly! Check $LOG_DIR/${NAME}_vllm.log"
            return 1
        fi
        sleep 2
    done
    if [ "$READY" = "0" ]; then
        echo "    vLLM timed out! Check $LOG_DIR/${NAME}_vllm.log"
        return 1
    fi

    # ── warmup ──
    echo "  [warmup] Running..."
    if WARMUP_ASR_PORT=$ASR_PORT WARMUP_TTS_PORT=$TTS_PORT WARMUP_VLLM_PORT=$VLLM_PORT \
       python -u warmup.py > "$LOG_DIR/${NAME}_warmup.log" 2>&1; then
        echo "    Warmup ok."
    else
        echo "    Warmup returned non-zero (services still usable)."
    fi

    echo "===== Instance $NAME fully started ====="
}

# ========================================================================
# 在某个实例上顺序跑一批视频
# 参数: <实例名> <VLLM_PORT> <数据子集名> <root> <output> [gt_root] [shard_index] [num_shards]
# ========================================================================
run_subset() {
    local NAME=$1
    local PORT=$2
    local SUBSET=$3
    local ROOT=$4
    local OUTPUT=$5
    local GT_ROOT=${6:-}
    local SHARD_IDX=${7:-0}
    local NUM_SHARDS=${8:-1}

    echo "[run] $NAME: $SUBSET (shard $SHARD_IDX/$NUM_SHARDS) -> $OUTPUT"
    local CMD=(
        python -u eval/run_dataset.py
        --root "$ROOT"
        --output "$OUTPUT"
        --port "$PORT"
        --num_shards "$NUM_SHARDS"
        --shard_index "$SHARD_IDX"
    )
    if [ -n "$SKIP_FLAG" ]; then
        CMD+=($SKIP_FLAG)
    fi
    if [ -n "$GT_ROOT" ]; then
        CMD+=(--gt_root "$GT_ROOT")
    fi

    "${CMD[@]}" >> "$LOG_DIR/${NAME}_${SUBSET}.log" 2>&1
    echo "[run] $NAME: $SUBSET shard $SHARD_IDX done."
}

# ========================================================================
# Main
# ========================================================================
echo "============================================"
echo "  AURA-8b Batch Inference"
echo "  Model:  $MODEL_PATH"
echo "  Output: $OUTPUT_ROOT"
echo "  Data:   $DATA_ROOT"
echo "============================================"

# ── 启动两个实例 ──
start_instance "A" $INST_A_GPU_ASR $INST_A_GPU_TTS "$INST_A_GPU_VLLM" \
               $INST_A_ASR_PORT $INST_A_TTS_PORT $INST_A_VLLM_PORT

start_instance "B" $INST_B_GPU_ASR $INST_B_GPU_TTS "$INST_B_GPU_VLLM" \
               $INST_B_ASR_PORT $INST_B_TTS_PORT $INST_B_VLLM_PORT

echo ""
echo "============================================"
echo "  Both instances running. Starting inference."
echo "============================================"

# ── 并行推理: 各实例负责所有视频的一半 ──
# 每个子集按 2 shards 切分；run_dataset.py 直接读 OmniInteract data/ 布局
# (video_json_map.json 或 videos_bench/ 镜像 annotations/)，无需 gt_root。
#   1q1a: 150 / 1q1a_math: 60 / 1qna: 40

mkdir -p "$OUTPUT_ROOT/1q1a" "$OUTPUT_ROOT/1q1a_math" "$OUTPUT_ROOT/1qna"

(
    run_subset "A" $INST_A_VLLM_PORT "1q1a" \
        "$DATA_ROOT/1q1a" "$OUTPUT_ROOT/1q1a" "" 0 2
    run_subset "A" $INST_A_VLLM_PORT "1q1a_math" \
        "$DATA_ROOT/1q1a_math" "$OUTPUT_ROOT/1q1a_math" "" 0 2
    run_subset "A" $INST_A_VLLM_PORT "1qna" \
        "$DATA_ROOT/1qna" "$OUTPUT_ROOT/1qna" "" 0 2
) &
PID_A=$!
ALL_PIDS+=($PID_A)

(
    run_subset "B" $INST_B_VLLM_PORT "1q1a" \
        "$DATA_ROOT/1q1a" "$OUTPUT_ROOT/1q1a" "" 1 2
    run_subset "B" $INST_B_VLLM_PORT "1q1a_math" \
        "$DATA_ROOT/1q1a_math" "$OUTPUT_ROOT/1q1a_math" "" 1 2
    run_subset "B" $INST_B_VLLM_PORT "1qna" \
        "$DATA_ROOT/1qna" "$OUTPUT_ROOT/1qna" "" 1 2
) &
PID_B=$!
ALL_PIDS+=($PID_B)

echo "[main] Instance A (PID $PID_A): shard 0/2 of 1q1a + 1q1a_math + 1qna"
echo "[main] Instance B (PID $PID_B): shard 1/2 of 1q1a + 1q1a_math + 1qna"
echo ""

# ── 等待推理完成 ──
FAIL=0
wait $PID_A || { echo "[main] Instance A inference FAILED (exit $?)"; FAIL=1; }
wait $PID_B || { echo "[main] Instance B inference FAILED (exit $?)"; FAIL=1; }

echo ""
echo "============================================"
echo "  Inference complete."
echo ""

# ── 统计结果 ──
N_1Q1A=$(find "$OUTPUT_ROOT/1q1a" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
N_1Q1A_MATH=$(find "$OUTPUT_ROOT/1q1a_math" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
N_1QNA=$(find "$OUTPUT_ROOT/1qna" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
echo "  1q1a outputs:      $N_1Q1A / 150"
echo "  1q1a_math outputs: $N_1Q1A_MATH / 60"
echo "  1qna outputs:      $N_1QNA / 40"
echo "  Total:             $((N_1Q1A + N_1Q1A_MATH + N_1QNA)) / 250"
echo ""
echo "  Logs: $LOG_DIR/"
echo "============================================"

if [ "$FAIL" -ne 0 ]; then
    echo "WARNING: Some inference processes failed. Check logs."
    exit 1
fi
