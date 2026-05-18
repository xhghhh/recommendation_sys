#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- DDP launcher ----
# Auto-detect visible GPUs and use torchrun to spawn one process per GPU.
# Falls back to single-process (no torchrun) when 0 or 1 GPU is visible.
#
# Override via env:
#   NPROC_PER_NODE=2 bash run.sh           # force 2 processes
#   NPROC_PER_NODE=1 bash run.sh           # force single-process (no DDP)
if [ -z "${NPROC_PER_NODE}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    else
        NPROC_PER_NODE=0
    fi
    if [ -z "${NPROC_PER_NODE}" ] || [ "${NPROC_PER_NODE}" = "0" ]; then
        NPROC_PER_NODE=1
    fi
fi
echo "[run.sh] NPROC_PER_NODE=${NPROC_PER_NODE}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
        "${SCRIPT_DIR}/train.py" \
        --ns_tokenizer_type rankmixer \
        --user_ns_tokens 5 \
        --item_ns_tokens 2 \
        --num_queries 2 \
        --ns_groups_json "" \
        --emb_skip_threshold 1000000 \
        --num_workers 8 \
        --d_model 128 \
        --emb_dim 128 \
        --num_hyformer_blocks 3 \
        --batch_size 128 \
        --use_rope \
        --loss_type focal \
        --focal_alpha 0.25 \
        --focal_gamma 2.0 \
        --valid_ratio 0.05 \
        --warmup_steps 1000 \
        --reinit_cardinality_threshold 50000 \
        --buffer_batches 50 \
        "$@"
else
    python3 -u "${SCRIPT_DIR}/train.py" \
        --ns_tokenizer_type rankmixer \
        --user_ns_tokens 5 \
        --item_ns_tokens 2 \
        --num_queries 2 \
        --ns_groups_json "" \
        --emb_skip_threshold 1000000 \
        --num_workers 8 \
        --d_model 128 \
        --emb_dim 128 \
        --num_hyformer_blocks 3 \
        --batch_size 128 \
        --use_rope \
        --loss_type focal \
        --focal_alpha 0.25 \
        --focal_gamma 2.0 \
        --valid_ratio 0.05 \
        --warmup_steps 1000 \
        --reinit_cardinality_threshold 50000 \
        --buffer_batches 50 \
        "$@"
fi

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
