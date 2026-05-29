# -*- coding: utf-8 -*-
""" Pre-aggregate whole_slice_attention_layer_L.* into compact CSV files for plotting - Output full aggregation(sum, mean) and optional compact TopK CSV - Support chunked reading for large CSV files;read Parquet input in one pass (can be changed to chunked mode if needed) - Gene view:kv_gene_symbol and q_gene_symbol - dimensions:spot-level(center×neighbor×gene), domain-level(neighbors: center×domain×gene;global: domain×gene) - metrics:attn_score of sum and mean (both are computed and saved) """

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
        raise ValueError("adata.obs  in not found ground_truth column")
    s = adata.obs["ground_truth"].astype(str).copy()
    s.index = s.index.astype(str)
    return s  # Series: spot_name -> domain

def stream_read(attn_path: str, usecols: Optional[List[str]] = None, chunksize: int = 2_000_000):
    """
    Stream-read the attention table:
    - Parquet: read directly into a DataFrame
    - CSV: use pandas chunksize for CSV chunks
    """
    if attn_path.endswith(".parquet"):
        df = pd.read_parquet(attn_path, columns=usecols)
        yield df
    else:
        for chunk in pd.read_csv(attn_path, usecols=usecols, chunksize=chunksize):
            yield chunk

# ---------------- :aggregation sum and count, after compute mean ----------------

def prepare_aggregations_csv(
    out_dir: str,
    layer: int = 5,
    value_col: str = "attn_score",
    chunksize: int = 2_000_000,
    keep_topk_for_speed: Optional[int] = None,   # 5 or 10;None meanstrim
):
    """ Generate compact CSV files for plotting,saved under out_dir/agg_csv/. """
    # Input paths
    attn_parquet = os.path.join(out_dir, f"whole_slice_attention_layer_{layer}.parquet")
    attn_csv = os.path.join(out_dir, f"whole_slice_attention_layer_{layer}.csv")
    if os.path.exists(attn_parquet):
        attn_path = attn_parquet
    elif os.path.exists(attn_csv):
        attn_path = attn_csv
    else:
        raise FileNotFoundError(f"not found whole_slice_attention_layer_{layer}  of  Parquet  or  CSV")

    adata_path = os.path.join(out_dir, "adata_with_metadata.h5ad")
    dom_series = load_domains(adata_path)

    # Output directory
    save_dir = os.path.join(out_dir, "agg_csv")
    os.makedirs(save_dir, exist_ok=True)

    usecols = ["center_name", "neighbor_name",
               "q_gene_symbol", "kv_gene_symbol",
               value_col]

    # (columntableaggregation, after concataggregation)
    acc_spot_kv = []   # center, neighbor, gene, sum, count
    acc_spot_q  = []

    acc_dom_nei_kv = []  # center, domain, gene, sum, count
    acc_dom_nei_q  = []

    acc_dom_glb_kv = []  # domain, gene, sum, count
    acc_dom_glb_q  = []

    # readingaggregation
    for chunk in stream_read(attn_path, usecols=usecols, chunksize=chunksize):
        # &
        for col in ["center_name","neighbor_name","q_gene_symbol","kv_gene_symbol"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str)

        # domain (for "NA")
        chunk["neighbor_domain"] = chunk["neighbor_name"].map(lambda x: dom_series.get(str(x), "NA"))

        # filtergene
        sub_kv = chunk[chunk["kv_gene_symbol"].str.len() > 0][["center_name","neighbor_name","neighbor_domain","kv_gene_symbol", value_col]].copy()
        sub_q  = chunk[chunk["q_gene_symbol"].str.len()  > 0][["center_name","neighbor_name","neighbor_domain","q_gene_symbol",  value_col]].copy()

        # spot-level:center × neighbor × gene
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

        # domain-level (neighbors):center × domain × gene
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

        # domain-level (global):domain × gene
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

    # aggregation(sum, count -> mean)
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

    # saveCSV
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

    # TopK CSV (optional)
    if keep_topk_for_speed is not None and keep_topk_for_speed > 0:
        k = int(keep_topk_for_speed)

        # spot-level: each center×neighbor dimensions gene of top K (sum and mean)
        def topk_spot(df: pd.DataFrame, by_cols: List[str], k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
            df = df.copy()
            df["rank_sum"]  = df.groupby(by_cols)["sum"].rank(method="first", ascending=False)
            df["rank_mean"] = df.groupby(by_cols)["mean"].rank(method="first", ascending=False)
            top_sum  = df[df["rank_sum"]  <= k].drop(columns=["rank_sum","rank_mean"])
            top_mean = df[df["rank_mean"] <= k].drop(columns=["rank_sum","rank_mean"])
            return top_sum, top_mean

        # domain-level (neighbors): each center×domain gene of topK
        def topk_domain_neighbors(df: pd.DataFrame, k: int):
            df = df.copy()
            df["rank_sum"]  = df.groupby(["center_name","domain"])["sum"].rank(method="first", ascending=False)
            df["rank_mean"] = df.groupby(["center_name","domain"])["mean"].rank(method="first", ascending=False)
            top_sum  = df[df["rank_sum"]  <= k].drop(columns=["rank_sum","rank_mean"])
            top_mean = df[df["rank_mean"] <= k].drop(columns=["rank_sum","rank_mean"])
            return top_sum, top_mean

        # domain-level (global): each domain gene of topK
        def topk_domain_global(df: pd.DataFrame, k: int):
            df = df.copy()
            df["rank_sum"]  = df.groupby(["domain"])["sum"].rank(method="first", ascending=False)
            df["rank_mean"] = df.groupby(["domain"])["mean"].rank(method="first", ascending=False)
            top_sum  = df[df["rank_sum"]  <= k].drop(columns=["rank_sum","rank_mean"])
            top_mean = df[df["rank_mean"] <= k].drop(columns=["rank_sum","rank_mean"])
            return top_sum, top_mean

        # compute topK
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

        # Save TopK CSV
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
    # for of exportdirectory and layer number
    out_dir = "./PDAC/whole_slice_data_20251028_173836"  # replace with the actual directory
    layer = 5
    # keep_topk_for_speed: 5;trim None
    prepare_aggregations_csv(out_dir=out_dir, layer=layer, value_col="attn_score",
                             chunksize=2_000_000, keep_topk_for_speed=5)