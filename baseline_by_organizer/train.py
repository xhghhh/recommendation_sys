"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Debug mode: use local demo_1000.parquet for quick testing.
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode: use local demo_1000.parquet for quick testing')
    parser.add_argument('--debug_data', type=str,
                        default='/Users/huanglichen/Desktop/recommendation_sys/demo_1000.parquet',
                        help='Path to demo parquet file (only used with --debug)')
    parser.add_argument('--debug_schema', type=str,
                        default='/Users/huanglichen/Desktop/recommendation_sys/demo_schema.json',
                        help='Path to demo schema.json (only used with --debug)')

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=1000,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    # User-item interaction & inter-sequence attention.
    parser.add_argument('--num_user_seqs', type=int, default=2,
                        help='Number of user-side sequence domains (the first N '
                             'domains in sorted order). Remaining domains are '
                             'treated as item-side. Used by UserItemCrossAttention.')
    parser.add_argument('--use_user_item_cross_attn', action='store_true', default=True,
                        help='Enable bidirectional user-item cross-attention '
                             'after the HyFormer block stack')
    parser.add_argument('--no_user_item_cross_attn', dest='use_user_item_cross_attn',
                        action='store_false',
                        help='Disable user-item cross-attention')
    parser.add_argument('--use_inter_seq_attn', action='store_true', default=True,
                        help='Enable inter-sequence self-attention over all Q tokens')
    parser.add_argument('--no_inter_seq_attn', dest='use_inter_seq_attn',
                        action='store_false',
                        help='Disable inter-sequence self-attention')

    # Learning rate scheduling.
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of linear warmup steps for the dense optimizer '
                             '(0 = no warmup)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch size = '
                             'batch_size * grad_accum_steps')
    parser.add_argument('--top_k_checkpoints', type=int, default=3,
                        help='Number of top-k best checkpoints to keep on disk '
                             '(ranked by validation AUC)')

    # User VQ regularization.
    parser.add_argument('--use_user_vq', action='store_true', default=False,
                        help='Enable Vector Quantization on user NS tokens '
                             'for regularization')
    parser.add_argument('--no_user_vq', dest='use_user_vq', action='store_false',
                        help='Disable user VQ')
    parser.add_argument('--vq_codebook_size', type=int, default=256,
                        help='Number of codebook vectors in the user VQ encoder')
    parser.add_argument('--vq_commitment_cost', type=float, default=0.25,
                        help='Commitment loss weight for user VQ '
                             '(higher = stronger regularization)')

    # Dense feature augmentation.
    parser.add_argument('--aug_dense_ratio', type=float, default=0.0,
                        help='Fraction of each training batch to augment via '
                             'dense feature perturbation (0.0 = no augmentation, '
                             '0.5 = augment 50% of rows, 1.0 = augment all rows)')
    parser.add_argument('--aug_dense_noise_std', type=float, default=0.1,
                        help='Gaussian noise std for dense feature augmentation')
    parser.add_argument('--aug_dense_scale_range', type=float, default=0.1,
                        help='Random scale range for dense feature augmentation '
                             '(scale ~ Uniform[1-range, 1+range])')

    args = parser.parse_args()

    # Debug mode overrides: use local demo data and small defaults.
    if args.debug:
        args.data_dir = args.debug_data
        args.schema_path = args.debug_schema
        args.batch_size = args.batch_size or 32
        args.num_epochs = min(args.num_epochs, 3)
        args.patience = min(args.patience, 2)
        args.num_workers = 0
        args.d_model = min(args.d_model, 32)
        args.num_hyformer_blocks = 1
        args.valid_ratio = 0.0  # demo has only 1 Row Group; skip validation split
        logging.info("DEBUG MODE: using local demo data with small model")

    # Environment variables take precedence (skip in debug mode).
    if not args.debug:
        args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
        args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
        # NOTE: platform does NOT provide TRAIN_LOG_PATH; fall back to a writable
        # platform-managed dir to avoid FileHandler crashing on read-only script dir.
        args.log_dir = (
            os.environ.get('TRAIN_LOG_PATH')
            or args.log_dir
            or os.environ.get('TRAIN_TF_EVENTS_PATH')
            or os.environ.get('USER_CACHE_PATH')
            or '/tmp'
        )
        args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    # Set default output dirs if still None.
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_ckpt')
    if args.log_dir is None:
        args.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_logs')
    if not hasattr(args, 'tf_events_dir') or args.tf_events_dir is None:
        args.tf_events_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_tb')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    # ---- DDP setup (auto-detect torchrun env) ----
    ddp_env = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ
               and 'LOCAL_RANK' in os.environ)
    if ddp_env:
        import torch.distributed as dist
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = f'cuda:{local_rank}'
            backend = 'nccl'
        else:
            args.device = 'cpu'
            backend = 'gloo'
        dist.init_process_group(backend=backend)
        logging.info(
            f"DDP enabled | rank={rank} local_rank={local_rank} "
            f"world_size={world_size} backend={backend} device={args.device}"
        )
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        logging.info("DDP disabled (single-process training)")
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    is_main = (rank == 0)

    # Only rank-0 writes TensorBoard events to avoid duplicate writers.
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir) if is_main else None

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    elif os.path.isdir(args.data_dir):
        schema_path = os.path.join(args.data_dir, 'schema.json')
    else:
        # data_dir is a single file; look for schema.json alongside it.
        schema_path = os.path.join(os.path.dirname(args.data_dir), 'schema.json')
        if not os.path.exists(schema_path):
            schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        rank=rank,
        world_size=world_size,
        aug_dense_ratio=args.aug_dense_ratio,
        aug_dense_noise_std=args.aug_dense_noise_std,
        aug_dense_scale_range=args.aug_dense_scale_range,
    )

    # ---- Dataset size diagnostics ----
    try:
        total_rows = getattr(pcvr_dataset, 'num_rows', None)
        train_steps = len(train_loader) if train_loader is not None else 0
        valid_steps = len(valid_loader) if valid_loader is not None else 0
        logging.info(
            f"Dataset size | total_rows={total_rows} | "
            f"batch_size={args.batch_size} | "
            f"train_steps_per_epoch={train_steps} | "
            f"valid_steps={valid_steps} | "
            f"max_epochs={args.num_epochs} | "
            f"theoretical_max_total_steps={train_steps * args.num_epochs}"
        )
    except Exception as e:
        logging.warning(f"Failed to log dataset size diagnostics: {e}")

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "num_user_seqs": args.num_user_seqs,
        "use_user_item_cross_attn": args.use_user_item_cross_attn,
        "use_inter_seq_attn": args.use_inter_seq_attn,
        "use_user_vq": args.use_user_vq,
        "vq_codebook_size": args.vq_codebook_size,
        "vq_commitment_cost": args.vq_commitment_cost,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info BEFORE DDP wrap (DDP hides custom attrs behind .module).
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # Wrap with DDP if launched via torchrun (>=2 processes).
    if ddp_env and world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters=True: PCVRHyFormer has conditional code paths
        # (e.g. domain-specific seq encoders) where some params may not
        # contribute to a given step's loss; setting True avoids DDP
        # 'unused parameter' errors at the small cost of an extra reduce.
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )
        logging.info(f"Wrapped model with DistributedDataParallel (device_ids=[{local_rank}])")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        rank=rank,
        world_size=world_size,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        top_k_checkpoints=args.top_k_checkpoints,
    )

    trainer.train()
    if writer is not None:
        writer.close()

    if ddp_env:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
