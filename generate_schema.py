"""
从 demo_1000.parquet 自动生成 schema.json
供 baseline_by_organizer/dataset.py 使用
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd


def infer_schema_from_parquet(parquet_path: str, output_path: str = None) -> dict:
    """
    从 Parquet 文件推断 schema.json 格式
    
    Args:
        parquet_path: Parquet 文件路径
        output_path: schema.json 输出路径（None 则不写文件）
    
    Returns:
        schema dict
    """
    print(f"读取 {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"  形状: {df.shape}")
    print(f"  列数: {len(df.columns)}")
    
    # ============ 分类列 ============
    user_int_cols = []    # [[fid, vocab_size, dim], ...]
    item_int_cols = []    # [[fid, vocab_size, dim], ...]
    user_dense_cols = []  # [[fid, dim], ...]
    seq_cfg = {}          # domain -> {prefix, ts_fid, features}
    
    # 解析列名，提取 fid 和类型
    user_int_pattern = re.compile(r'^user_int_feats_(\d+)$')
    item_int_pattern = re.compile(r'^item_int_feats_(\d+)$')
    user_dense_pattern = re.compile(r'^user_dense_feats_(\d+)$')
    seq_pattern = re.compile(r'^domain_([a-z])_seq_(\d+)$')
    
    for col in df.columns:
        # User Int
        m = user_int_pattern.match(col)
        if m:
            fid = int(m.group(1))
            dtype = df[col].dtype
            first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            
            if first_valid is not None and isinstance(first_valid, (list, np.ndarray)):
                # 数组型特征
                lengths = df[col].dropna().apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
                dim = int(lengths.max())
                # 计算词表大小
                all_vals = []
                for v in df[col].dropna():
                    if isinstance(v, (list, np.ndarray)):
                        all_vals.extend([int(x) for x in v if x > 0])
                vocab_size = max(all_vals) + 1 if all_vals else 1
            else:
                # 标量型特征
                dim = 1
                vals = df[col].dropna().values
                vals = vals[vals > 0] if len(vals) > 0 else np.array([0])
                vocab_size = int(vals.max()) + 1 if len(vals) > 0 else 1
            
            user_int_cols.append([fid, vocab_size, dim])
            continue
        
        # Item Int
        m = item_int_pattern.match(col)
        if m:
            fid = int(m.group(1))
            first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            
            if first_valid is not None and isinstance(first_valid, (list, np.ndarray)):
                lengths = df[col].dropna().apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
                dim = int(lengths.max())
                all_vals = []
                for v in df[col].dropna():
                    if isinstance(v, (list, np.ndarray)):
                        all_vals.extend([int(x) for x in v if x > 0])
                vocab_size = max(all_vals) + 1 if all_vals else 1
            else:
                dim = 1
                vals = df[col].dropna().values
                vals = vals[vals > 0] if len(vals) > 0 else np.array([0])
                vocab_size = int(vals.max()) + 1 if len(vals) > 0 else 1
            
            item_int_cols.append([fid, vocab_size, dim])
            continue
        
        # User Dense
        m = user_dense_pattern.match(col)
        if m:
            fid = int(m.group(1))
            first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if first_valid is not None and isinstance(first_valid, (list, np.ndarray)):
                dim = len(first_valid)
            else:
                dim = 1
            user_dense_cols.append([fid, dim])
            continue
        
        # Sequence
        m = seq_pattern.match(col)
        if m:
            domain = f"seq_{m.group(1)}"
            fid = int(m.group(2))
            prefix = f"domain_{m.group(1)}_seq"
            
            if domain not in seq_cfg:
                seq_cfg[domain] = {
                    "prefix": prefix,
                    "ts_fid": None,
                    "features": []
                }
            
            # 计算词表大小
            all_vals = []
            for v in df[col].dropna():
                if isinstance(v, (list, np.ndarray)):
                    all_vals.extend([int(x) for x in v if x > 0])
            vocab_size = max(all_vals) + 1 if all_vals else 1
            
            seq_cfg[domain]["features"].append([fid, vocab_size])
            continue
    
    # ============ 推断时间戳列 ============
    # 序列中fid最大的通常是时间戳列（值域最大的int序列）
    for domain, cfg in seq_cfg.items():
        # 找到值域最大的特征作为 ts_fid
        max_vs_fid = None
        max_vs = 0
        for fid, vs in cfg["features"]:
            if vs > max_vs:
                max_vs = vs
                max_vs_fid = fid
        if max_vs > 1000000:  # 时间戳的值域通常很大
            cfg["ts_fid"] = max_vs_fid
    
    # ============ 排序 ============
    user_int_cols.sort(key=lambda x: x[0])
    item_int_cols.sort(key=lambda x: x[0])
    user_dense_cols.sort(key=lambda x: x[0])
    for domain in seq_cfg:
        seq_cfg[domain]["features"].sort(key=lambda x: x[0])
    
    schema = {
        "user_int": user_int_cols,
        "item_int": item_int_cols,
        "user_dense": user_dense_cols,
        "seq": seq_cfg
    }
    
    # ============ 打印统计 ============
    print(f"\n=== Schema 统计 ===")
    print(f"User Int 特征: {len(user_int_cols)} 个")
    user_int_dim = sum(x[2] for x in user_int_cols)
    print(f"  总维度: {user_int_dim}")
    
    print(f"Item Int 特征: {len(item_int_cols)} 个")
    item_int_dim = sum(x[2] for x in item_int_cols)
    print(f"  总维度: {item_int_dim}")
    
    print(f"User Dense 特征: {len(user_dense_cols)} 个")
    user_dense_dim = sum(x[1] for x in user_dense_cols)
    print(f"  总维度: {user_dense_dim}")
    
    print(f"序列域: {list(seq_cfg.keys())}")
    for domain, cfg in seq_cfg.items():
        n_feats = len(cfg["features"])
        ts_fid = cfg.get("ts_fid")
        print(f"  {domain}: {n_feats} 个特征, ts_fid={ts_fid}")
    
    # ============ 写文件 ============
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"\nSchema 已保存到: {output_path}")
    
    return schema


if __name__ == '__main__':
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/huanglichen/Desktop/recommendation_sys/demo_1000.parquet'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/Users/huanglichen/Desktop/recommendation_sys/demo_schema.json'
    
    schema = infer_schema_from_parquet(parquet_path, output_path)
