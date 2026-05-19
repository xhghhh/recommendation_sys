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
        --d_model 96 \
        --emb_dim 96 \
        --num_hyformer_blocks 2 \
        --batch_size 256 \
        --use_rope \
        --loss_type focal \
        --focal_alpha 0.25 \
        --focal_gamma 2.0 \
        --valid_ratio 0.1 \
        --lr 1e-6 \
        --sparse_lr 0.01 \
        --warmup_steps 2000 \
        --dropout_rate 0.06 \
        --reinit_cardinality_threshold 50000 \
        --buffer_batches 50 \
        --use_user_vq \
        --vq_codebook_size 256 \
        --vq_commitment_cost 0.25 \
        --aug_dense_ratio 0.2 \
        --aug_dense_noise_std 0.1 \
        --aug_dense_scale_range 0.1 \
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
        --d_model 96 \
        --emb_dim 96 \
        --num_hyformer_blocks 2 \
        --batch_size 256 \
        --use_rope \
        --loss_type focal \
        --focal_alpha 0.25 \
        --focal_gamma 2.0 \
        --valid_ratio 0.1 \
        --lr 5e-5 \
        --sparse_lr 0.01 \
        --warmup_steps 2000 \
        --dropout_rate 0.05 \
        --reinit_cardinality_threshold 50000 \
        --buffer_batches 50 \
        --use_user_vq \
        --vq_codebook_size 256 \
        --vq_commitment_cost 0.25 \
        --aug_dense_ratio 0.5 \
        --aug_dense_noise_std 0.1 \
        --aug_dense_scale_range 0.1 \
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
