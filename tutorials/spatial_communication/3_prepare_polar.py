# -*- coding: utf-8 -*-
"""
将 whole_slice_attention_layer_L.* 预聚合为绘图所需的紧凑CSV
- 输出全量聚合(含 sum、mean)与可选的TopK精简CSV
- 支持分块读取大型CSV；若输入为Parquet则一次性读取（可按需修改为分块）
- 基因视角：kv_gene_symbol 与 q_gene_symbol
- 维度：spot-level(center×neighbor×gene)、domain-level(neighbors: center×domain×gene；global: domain×gene)
- 指标：attn_score 的 sum 与 mean（均计算并保存）
"""

import os
import gc
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import anndata as ad

# ---------------- I/O ----------------

def load_domains(adata_path: str) -> pd.Series:
    adata = ad.read_h5ad(adata_path)
    if "ground_truth" not in adata.obs:
        raise ValueError("adata.obs 中未找到 ground_truth 列")
    s = adata.obs["ground_truth"].astype(str).copy()
    s.index = s.index.astype(str)
    return s  # Series: spot_name -> domain

def stream_read(attn_path: str, usecols: Optional[List[str]] = None, chunksize: int = 2_000_000):
    """
    流式读取注意力表：
    - Parquet: 直接一次性读取为 DataFrame
    - CSV: 使用 pandas chunksize 分块
    """
    if attn_path.endswith(".parquet"):
        df = pd.read_parquet(attn_path, columns=usecols)
        yield df
    else:
        for chunk in pd.read_csv(attn_path, usecols=usecols, chunksize=chunksize):
            yield chunk

# ---------------- 主流程：分块聚合 sum 与 count，最后统一计算 mean ----------------

def prepare_aggregations_csv(
    out_dir: str,
    layer: int = 5,
    value_col: str = "attn_score",
    chunksize: int = 2_000_000,
    keep_topk_for_speed: Optional[int] = None,   # 例如 5 或 10；None 表示不裁剪
):
    """
    生成用于绘图的紧凑CSV文件，保存于 out_dir/agg_csv/ 下。
    """
    # 输入路径
    attn_parquet = os.path.join(out_dir, f"whole_slice_attention_layer_{layer}.parquet")
    attn_csv = os.path.join(out_dir, f"whole_slice_attention_layer_{layer}.csv")
    if os.path.exists(attn_parquet):
        attn_path = attn_parquet
    elif os.path.exists(attn_csv):
        attn_path = attn_csv
    else:
        raise FileNotFoundError(f"未找到 whole_slice_attention_layer_{layer} 的 Parquet 或 CSV")

    adata_path = os.path.join(out_dir, "adata_with_metadata.h5ad")
    dom_series = load_domains(adata_path)

    # 输出目录
    save_dir = os.path.join(out_dir, "agg_csv")
    os.makedirs(save_dir, exist_ok=True)

    usecols = ["center_name", "neighbor_name",
               "q_gene_symbol", "kv_gene_symbol",
               value_col]

    # 累加容器（用列表存块聚合结果，最后concat再二次聚合）
    acc_spot_kv = []   # center, neighbor, gene, sum, count
    acc_spot_q  = []

    acc_dom_nei_kv = []  # center, domain, gene, sum, count
    acc_dom_nei_q  = []

    acc_dom_glb_kv = []  # domain, gene, sum, count
    acc_dom_glb_q  = []

    # 分块读取并聚合
    for chunk in stream_read(attn_path, usecols=usecols, chunksize=chunksize):
        # 统一类型 & 清洗
        for col in ["center_name","neighbor_name","q_gene_symbol","kv_gene_symbol"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str)

        # 映射邻居domain（无匹配设为 "NA"）
        chunk["neighbor_domain"] = chunk["neighbor_name"].map(lambda x: dom_series.get(str(x), "NA"))

        # 过滤空基因
        sub_kv = chunk[chunk["kv_gene_symbol"].str.len() > 0][["center_name","neighbor_name","neighbor_domain","kv_gene_symbol", value_col]].copy()
        sub_q  = chunk[chunk["q_gene_symbol"].str.len()  > 0][["center_name","neighbor_name","neighbor_domain","q_gene_symbol",  value_col]].copy()

        # spot-level：center × neighbor × gene
        g = sub_kv.groupby(["center_name","neighbor_name","kv_gene_symbol"], observed=True)[value_col]
        acc_spot_kv.append(
            g.agg(sum_val="sum", cnt="count").reset_index()
             .rename(columns={"kv_gene_symbol":"gene","sum_val":"sum","cnt":"count"})
        )
        g = sub_q.groupby(["center_name","neighbor_name","q_gene_symbol"], observed=True)[value_col]
        acc_spot_q.append(
            g.agg(sum_val="sum", cnt="count").reset_index()
             .rename(columns={"q_gene_symbol":"gene","sum_val":"sum","cnt":"count"})
        )

        # domain-level（neighbors 范围）：center × domain × gene
        g = sub_kv.groupby(["center_name","neighbor_domain","kv_gene_symbol"], observed=True)[value_col]
        acc_dom_nei_kv.append(
            g.agg(sum_val="sum", cnt="count").reset_index()
             .rename(columns={"neighbor_domain":"domain","kv_gene_symbol":"gene","sum_val":"sum","cnt":"count"})
        )
        g = sub_q.groupby(["center_name","neighbor_domain","q_gene_symbol"], observed=True)[value_col]
        acc_dom_nei_q.append(
            g.agg(sum_val="sum", cnt="count").reset_index()
             .rename(columns={"neighbor_domain":"domain","q_gene_symbol":"gene","sum_val":"sum","cnt":"count"})
        )

        # domain-level（global）：domain × gene
        g = sub_kv.groupby(["neighbor_domain","kv_gene_symbol"], observed=True)[value_col]
        acc_dom_glb_kv.append(
            g.agg(sum_val="sum", cnt="count").reset_index()
             .rename(columns={"neighbor_domain":"domain","kv_gene_symbol":"gene","sum_val":"sum","cnt":"count"})
        )
        g = sub_q.groupby(["neighbor_domain","q_gene_symbol"], observed=True)[value_col]
        acc_dom_glb_q.append(
            g.agg(sum_val="sum", cnt="count").reset_index()
             .rename(columns={"neighbor_domain":"domain","q_gene_symbol":"gene","sum_val":"sum","cnt":"count"})
        )

        del sub_kv, sub_q, chunk
        gc.collect()

    # 合并并做最终聚合(sum、count -> mean)
    def combine_and_finalize(parts: List[pd.DataFrame], key_cols: List[str]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=key_cols+["sum","count","mean"])
        df = pd.concat(parts, ignore_index=True)
        df = df.groupby(key_cols, observed=True).agg(sum=("sum","sum"),
                                                     count=("count","sum")).reset_index()
        df["mean"] = df["sum"] / df["count"].replace(0, np.nan)
        df["mean"] = df["mean"].fillna(0.0)
        return df

    spot_kv = combine_and_finalize(acc_spot_kv, ["center_name","neighbor_name","gene"])
    spot_q  = combine_and_finalize(acc_spot_q,  ["center_name","neighbor_name","gene"])

    dom_nei_kv = combine_and_finalize(acc_dom_nei_kv, ["center_name","domain","gene"])
    dom_nei_q  = combine_and_finalize(acc_dom_nei_q,  ["center_name","domain","gene"])

    dom_glb_kv = combine_and_finalize(acc_dom_glb_kv, ["domain","gene"])
    dom_glb_q  = combine_and_finalize(acc_dom_glb_q,  ["domain","gene"])

    # 保存全量CSV
    agg_dir = os.path.join(out_dir, "agg_csv")
    os.makedirs(agg_dir, exist_ok=True)

    paths_full = {
        "spot_kv":      os.path.join(agg_dir, "spot_level_kv.csv"),
        "spot_q":       os.path.join(agg_dir, "spot_level_q.csv"),
        "dom_nei_kv":   os.path.join(agg_dir, "domain_level_kv_neighbors.csv"),
        "dom_nei_q":    os.path.join(agg_dir, "domain_level_q_neighbors.csv"),
        "dom_glb_kv":   os.path.join(agg_dir, "domain_level_kv_global.csv"),
        "dom_glb_q":    os.path.join(agg_dir, "domain_level_q_global.csv"),
    }

    spot_kv.to_csv(paths_full["spot_kv"], index=False)
    spot_q.to_csv(paths_full["spot_q"], index=False)
    dom_nei_kv.to_csv(paths_full["dom_nei_kv"], index=False)
    dom_nei_q.to_csv(paths_full["dom_nei_q"], index=False)
    dom_glb_kv.to_csv(paths_full["dom_glb_kv"], index=False)
    dom_glb_q.to_csv(paths_full["dom_glb_q"], index=False)

    print("Saved full aggregation CSVs:")
    for k, p in paths_full.items():
        print(f" - {k}: {p}")

    # 生成 TopK 精简CSV（可选）
    if keep_topk_for_speed is not None and keep_topk_for_speed > 0:
        k = int(keep_topk_for_speed)

        # spot-level：每个 center×neighbor 维度取 gene 的 topK（分别就 sum 和 mean 选）
        def topk_spot(df: pd.DataFrame, by_cols: List[str], k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
            df = df.copy()
            df["rank_sum"]  = df.groupby(by_cols)["sum"].rank(method="first", ascending=False)
            df["rank_mean"] = df.groupby(by_cols)["mean"].rank(method="first", ascending=False)
            top_sum  = df[df["rank_sum"]  <= k].drop(columns=["rank_sum","rank_mean"])
            top_mean = df[df["rank_mean"] <= k].drop(columns=["rank_sum","rank_mean"])
            return top_sum, top_mean

        # domain-level（neighbors 范围）：每个 center×domain 取 gene 的 topK
        def topk_domain_neighbors(df: pd.DataFrame, k: int):
            df = df.copy()
            df["rank_sum"]  = df.groupby(["center_name","domain"])["sum"].rank(method="first", ascending=False)
            df["rank_mean"] = df.groupby(["center_name","domain"])["mean"].rank(method="first", ascending=False)
            top_sum  = df[df["rank_sum"]  <= k].drop(columns=["rank_sum","rank_mean"])
            top_mean = df[df["rank_mean"] <= k].drop(columns=["rank_sum","rank_mean"])
            return top_sum, top_mean

        # domain-level（global）：每个 domain 取 gene 的 topK
        def topk_domain_global(df: pd.DataFrame, k: int):
            df = df.copy()
            df["rank_sum"]  = df.groupby(["domain"])["sum"].rank(method="first", ascending=False)
            df["rank_mean"] = df.groupby(["domain"])["mean"].rank(method="first", ascending=False)
            top_sum  = df[df["rank_sum"]  <= k].drop(columns=["rank_sum","rank_mean"])
            top_mean = df[df["rank_mean"] <= k].drop(columns=["rank_sum","rank_mean"])
            return top_sum, top_mean

        # 计算 topK
        tk_spot_kv_sum,  tk_spot_kv_mean  = topk_spot(spot_kv, ["center_name","neighbor_name"], k)
        tk_spot_q_sum,   tk_spot_q_mean   = topk_spot(spot_q,  ["center_name","neighbor_name"], k)

        tk_dom_nei_kv_sum, tk_dom_nei_kv_mean = topk_domain_neighbors(dom_nei_kv, k)
        tk_dom_nei_q_sum,  tk_dom_nei_q_mean  = topk_domain_neighbors(dom_nei_q,  k)

        tk_dom_glb_kv_sum, tk_dom_glb_kv_mean = topk_domain_global(dom_glb_kv, k)
        tk_dom_glb_q_sum,  tk_dom_glb_q_mean  = topk_domain_global(dom_glb_q,  k)

        paths_topk = {
            "spot_kv_topk_sum":   os.path.join(agg_dir, f"spot_level_kv_top{k}_by_sum.csv"),
            "spot_kv_topk_mean":  os.path.join(agg_dir, f"spot_level_kv_top{k}_by_mean.csv"),
            "spot_q_topk_sum":    os.path.join(agg_dir, f"spot_level_q_top{k}_by_sum.csv"),
            "spot_q_topk_mean":   os.path.join(agg_dir, f"spot_level_q_top{k}_by_mean.csv"),

            "dom_nei_kv_topk_sum":  os.path.join(agg_dir, f"domain_level_kv_neighbors_top{k}_by_sum.csv"),
            "dom_nei_kv_topk_mean": os.path.join(agg_dir, f"domain_level_kv_neighbors_top{k}_by_mean.csv"),
            "dom_nei_q_topk_sum":   os.path.join(agg_dir, f"domain_level_q_neighbors_top{k}_by_sum.csv"),
            "dom_nei_q_topk_mean":  os.path.join(agg_dir, f"domain_level_q_neighbors_top{k}_by_mean.csv"),

            "dom_glb_kv_topk_sum":  os.path.join(agg_dir, f"domain_level_kv_global_top{k}_by_sum.csv"),
            "dom_glb_kv_topk_mean": os.path.join(agg_dir, f"domain_level_kv_global_top{k}_by_mean.csv"),
            "dom_glb_q_topk_sum":   os.path.join(agg_dir, f"domain_level_q_global_top{k}_by_sum.csv"),
            "dom_glb_q_topk_mean":  os.path.join(agg_dir, f"domain_level_q_global_top{k}_by_mean.csv"),
        }

        # 保存 TopK CSV
        tk_spot_kv_sum.to_csv(paths_topk["spot_kv_topk_sum"], index=False)
        tk_spot_kv_mean.to_csv(paths_topk["spot_kv_topk_mean"], index=False)
        tk_spot_q_sum.to_csv(paths_topk["spot_q_topk_sum"], index=False)
        tk_spot_q_mean.to_csv(paths_topk["spot_q_topk_mean"], index=False)

        tk_dom_nei_kv_sum.to_csv(paths_topk["dom_nei_kv_topk_sum"], index=False)
        tk_dom_nei_kv_mean.to_csv(paths_topk["dom_nei_kv_topk_mean"], index=False)
        tk_dom_nei_q_sum.to_csv(paths_topk["dom_nei_q_topk_sum"], index=False)
        tk_dom_nei_q_mean.to_csv(paths_topk["dom_nei_q_topk_mean"], index=False)

        tk_dom_glb_kv_sum.to_csv(paths_topk["dom_glb_kv_topk_sum"], index=False)
        tk_dom_glb_kv_mean.to_csv(paths_topk["dom_glb_kv_topk_mean"], index=False)
        tk_dom_glb_q_sum.to_csv(paths_topk["dom_glb_q_topk_sum"], index=False)
        tk_dom_glb_q_mean.to_csv(paths_topk["dom_glb_q_topk_mean"], index=False)

        print("Saved TopK compact CSVs:")
        for k, p in paths_topk.items():
            print(f" - {k}: {p}")

if __name__ == "__main__":
    # 修改为你的导出目录与层号
    out_dir = "./PDAC/whole_slice_data_20251028_173836"  # 替换为实际目录
    layer = 5
    # keep_topk_for_speed：例如 5；若不想裁剪则设 None
    prepare_aggregations_csv(out_dir=out_dir, layer=layer, value_col="attn_score",
                             chunksize=2_000_000, keep_topk_for_speed=5)