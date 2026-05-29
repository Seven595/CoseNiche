#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-attention export and enrichment preparation.

This script:
1. Exports long tables for each cluster with spot, gene, and partner entries.
2. Aggregates attention patterns for each domain or cluster.
3. Generates top partner gene tables for downstream enrichment analysis.

Outputs:
- `result_output/hbrc_layer{N}_clusters/`: detailed long-table CSV files.
- `result_output/enrichment_prepared/domain_tables/`: domain aggregation tables.
- `result_output/enrichment_prepared/all_domains_top_partners.csv`: top partner summary.
"""

import os
import pickle
from typing import List, Dict, Optional, Sequence, Tuple, Any
import matplotlib
matplotlib.use("Agg")  # Note.
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
import json
import csv
import re
import sys
from collections import defaultdict
import unicodedata

# ------------------ path (HBRC) ------------------
# base_dir = "/home/junning/projectnvme/ST/project-20-contrast-organ/Inference_embeddings/HBRC_clsrecon_nosemantic_finetune1"
# h5ad_path = '/home/junning/projectnvme/ST/h5ad/HBRC/human-breast-cancer.h5ad'
# truth_path =  '/home/junning/projectnvme/ST/Data/HBRC/hbrc_truth.csv'

# result_output_dir = "./HBRC1/result_output"
# enrichment_prepared_dir = "./HBRC1/result_output/enrichment_prepared"




#   ======== PDAC ======== 
base_dir = "/home/junning/projectnvme/ST/project-20-contrast-organ/Inference_embeddings/PDAC/PDAC_clsrecon_nosemantic_fintuneembedding_cls2genes_submodel"
h5ad_path = '/home/junning/projectnvme/ST/Data/PDAC/pdac.h5ad'
truth_path =  '/home/junning/projectnvme/ST/Data/PDAC/PDAC_truth.csv'

result_output_dir = "./PDAC/result_output"
enrichment_prepared_dir = "./PDAC/result_output/enrichment_prepared"

# Number of top genes exported for each domain.
top_kv_per_domain = 100

vocab_path = "/home/junning/projectnvme/ST/Data/Get_Embedding/vocab.json"
ctx_genes_pkl_path = os.path.join(base_dir, "context_genes.pkl")
attn_pkl_path = os.path.join(base_dir, "context_attention_scores.pkl")

# Select the attention layer for analysis.
use_layer = 5        # can
# Spot index used for debugging CSV output.
spot_idx_debug = 5

# ------------------ Gene filtering ------------------
# Whitelist prefixes such as HLA- and MIR are handled in the filtering logic.
# Prefix blacklist.
EXTRA_EXCLUDE_PREFIXES = ("AC", "AL", "LINC", "RP", "SNOR", "SCARNA")
DROP_WITH_DOT_OR_DASH = True
EXTRA_EXCLUDE_REGEX = None  # Optional custom exclusion regex.
DOT_DASH_PATTERN = re.compile(r"[.\-\u00B7\u2219\u2010\u2011\u2012\u2013\u2014\u2015]")

# ------------------ Helpers ------------------
def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.asarray(x)

def clean_symbol(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # Unicode / in to ASCII
    s = (s.replace("\u2010", "-")
           .replace("\u2011", "-")
           .replace("\u2012", "-")
           .replace("\u2013", "-")
           .replace("\u2014", "-")
           .replace("\u2015", "-")
           .replace("\u00B7", ".")
           .replace("\u2219", "."))
    s = " ".join(s.split())
    return s

def ids_to_clean_symbols(gids: Sequence[Any], id2sym: Dict[int, str]) -> List[str]:
    out = []
    for g in gids:
        if isinstance(g, str):
            sym = clean_symbol(g)
        else:
            try:
                sym = clean_symbol(id2sym.get(int(g), str(int(g))))
            except Exception:
                sym = clean_symbol(str(g))
        out.append(sym)
    return out

def build_id2sym(vd: Dict[Any, Any]) -> Dict[int, str]:
    # for {symbol: id}
    id2sym_try1 = {}
    bad = 0
    for k, v in vd.items():
        try:
            id2sym_try1[int(v)] = str(k).strip()
        except Exception:
            bad += 1
    if bad > 0 and bad > len(vd) * 0.5:
        # for {id: symbol}
        id2sym_try2 = {}
        for k, v in vd.items():
            try:
                id2sym_try2[int(k)] = str(v).strip()
            except Exception:
                pass
        if not id2sym_try2:
            raise RuntimeError(" from vocab.json id2sym;please checkfile.")
        return id2sym_try2
    if not id2sym_try1:
        raise RuntimeError("vocab is empty;please checkfile.")
    return id2sym_try1

def filter_symbols_and_attention(
    symbols: Sequence[str],
    A: np.ndarray,
    drop_with_dot_or_dash: bool = True,
    extra_exclude_prefixes: Sequence[str] = EXTRA_EXCLUDE_PREFIXES,
    extra_exclude_regex: Optional[str] = EXTRA_EXCLUDE_REGEX,
    whitelist: Optional[set] = None
) -> Tuple[List[str], np.ndarray, List[int]]:
    """ filtergene symbols and (by gene filteringplot) Step1: check - HLA- and MIR of gene Step2: check - of gene Step3: check - of gene Step4: Prefix blacklistCheck - before of gene Step5: check - of """
    L = len(symbols)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")
    # align L
    if A.shape[0] != L:
        A = A[:L, :L]
    prefixes = tuple(p.lower() for p in (extra_exclude_prefixes or []))
    kept = []
    for i, s in enumerate(symbols):
        s_clean = clean_symbol(s)

        # Step1: check - HLA- and MIR of gene
        if s_clean.startswith(('HLA-', 'MIR')):
            kept.append(i)
            continue

        # Step2: check - of gene
        if "." in s_clean:
            continue

        # Step3: check - filtergene, RNA
        regex_patterns = [
            r'^[A-Z]{2}\d+\.\d+$',      # AC012345.1 (gene)
            r'^ABBA\d+\.\d+$',           # ABBA01000934.1
            r'-AS\d*$',                  # A1BG-AS1 (RNA)
            r'^RP\d+-\w+\.\d+$',         # RP11-206L10.2 (gene)
            r'^CTD-\w+\.\d+$',           # CTDgene
            r'^CTC-\w+\.\d+$'            # CTCgene
        ]
        if any(re.match(pattern, s_clean) for pattern in regex_patterns):
            continue

        # Step4: Prefix blacklistCheck
        if prefixes and any(s_clean.lower().startswith(px) for px in prefixes):
            continue

        # Step5: check (if)
        if extra_exclude_regex:
            pat = re.compile(extra_exclude_regex)
            if pat.search(s_clean):
                continue

        kept.append(i)

    if len(kept) == 0:
        # :
        diag = np.diag(A)
        kept = [int(np.nanargmax(diag))] if diag.size > 0 else [0]
    kept_idx = np.asarray(kept, dtype=int)
    A_filt = A[np.ix_(kept_idx, kept_idx)]
    symbols_filt = [clean_symbol(symbols[i]) for i in kept_idx.tolist()]
    return symbols_filt, A_filt, kept_idx.tolist()

def assert_no_forbidden_symbols(symbols: Sequence[str],
                                prefixes: Sequence[str] = EXTRA_EXCLUDE_PREFIXES,
                                drop_dot_dash: bool = DROP_WITH_DOT_OR_DASH):
    """ after filtering of gene symbols (gene) """
    bad = []
    for s in symbols:
        s_clean = clean_symbol(s)
        
        # Skippinggene of Check
        if s_clean.startswith(('HLA-', 'MIR')):
            continue
            
        # Check
        if "." in s_clean:
            bad.append(("dot", s_clean, repr(s)))
        
        # CheckPrefix blacklist
        if any(s_clean.lower().startswith(px.lower()) for px in prefixes):
            bad.append(("prefix", s_clean, repr(s)))
    
    if bad:
        print("[ERROR] after filtering to invalid symbols (showing at most the first 30):")
        for kind, s_clean, r in bad[:30]:
            print(" -", kind, ":", s_clean, "raw_repr:", r)
        raise AssertionError("filterfailed:still containsinvalid symbols.")

def topk_partners_per_symbol(A_sym: np.ndarray, symbols: Sequence[str], K: int = 20, keep_self: bool = False):
    L = len(symbols)
    if A_sym.ndim != 2 or A_sym.shape[0] < L or A_sym.shape[1] < L:
        raise ValueError(f"A_sym shape mismatch: {A_sym.shape} vs L={L}")
    results = []
    for i in range(L):
        row = np.asarray(A_sym[i], dtype=float).copy()
        row[~np.isfinite(row)] = -np.inf
        if not keep_self:
            row[i] = -np.inf
        k = min(K, max(row.size - 1, 0)) if not keep_self else min(K, row.size)
        if k <= 0:
            results.append([])
            continue
        top_idx = np.argpartition(-row, kth=k-1)[:k] if k < row.size else np.arange(row.size, dtype=int)
        order = np.argsort(-row[top_idx])
        top_idx = top_idx[order]
        partners = [(symbols[j], float(row[j])) for j in top_idx if np.isfinite(row[j])]
        results.append(partners)
    return results, list(symbols)

def plot_sym_lower_triangle(Asym: np.ndarray, symbols: Sequence[str],
                            use_minmax: bool=True, vmin: Optional[float]=None, vmax: Optional[float]=None,
                            dpi: int=300, inches_per_50: float=15, max_fig_inches: float=10.0,
                            tick_max: int=50, tick_size: int=4, show_all_ticks: bool=False,
                            cmap: str="viridis", title: str="Symmetrized attention (lower triangle)",
                            save_path: Optional[str]=None):
    L = len(symbols)
    V = Asym.copy().astype(float)
    V[np.triu_indices(L, k=1)] = np.nan
    valid = np.isfinite(V)
    if not np.any(valid):
        print("No valid entries to plot."); return
    X = np.empty_like(V); X[:] = np.nan
    vals = V[valid]
    if use_minmax:
        if vmin is None: vmin = float(vals.min())
        if vmax is None: vmax = float(vals.max())
        if vmax <= vmin: vmax = vmin + 1e-12
        vals = np.clip(vals, vmin, vmax)
        X[valid] = (vals - vmin) / (vmax - vmin)
    else:
        X[valid] = vals
    fig_size = min(max_fig_inches, max(4.0, (L / 50.0) * inches_per_50))
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(X, aspect="equal", interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("min-max scaled attention (0-1)" if use_minmax else "raw attention")
    ax.set_title(title); ax.set_xlabel("Key/Value"); ax.set_ylabel("Query")
    ax.set_xlim(-0.5, L - 0.5); ax.set_ylim(L - 0.5, -0.5)
    def pick_ticks(Ln: int, max_n: int):
        if show_all_ticks: return list(range(Ln))
        step = max(1, Ln // max_n); return list(range(0, Ln, step))
    xt = pick_ticks(L, tick_max); yt = pick_ticks(L, tick_max)
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([symbols[x] for x in xt], rotation=90, fontsize=tick_size)
    ax.set_yticklabels([symbols[y] for y in yt], fontsize=tick_size)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(save_path.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)

# ------------------ by cluster of export (long table) and aggregation ------------------
def safe_filename(s: str, repl: str = "_") -> str:
    """ for availablefile of."""
    if s is None:
        return "NA"
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = re.sub(r"[\\/:\*\?\"<>\|\s]+", repl, s)
    s = s.strip(repl)
    return s or "NA"

def aggregate_domain_partners(df_long: pd.DataFrame) -> pd.DataFrame:
    """ single domain of tablerowsaggregation: - level1 (spot):partner_symbol in each spot of score_sum_by_spot = sum(score) - level2 (domain): partner_symbol of score_sum_by_spot and in column: partner_symbol, hit_spots, sum_strength, avg_strength in : hit_spots = of different spot sum_strength = all spot of score_sum_by_spot and avg_strength = sum_strength / hit_spots """
    # level1:spot total
    g1 = (df_long.groupby(["spot_idx", "partner_symbol"], as_index=False)["score"]
                 .sum()
                 .rename(columns={"score": "score_sum_by_spot"}))

    # level2:domain aggregation
    g2 = (g1.groupby("partner_symbol", as_index=False)
             .agg(hit_spots=("spot_idx", "nunique"),
                  sum_strength=("score_sum_by_spot", "sum")))

    g2["avg_strength"] = g2["sum_strength"] / g2["hit_spots"].replace(0, np.nan)
    g2["avg_strength"] = g2["avg_strength"].fillna(0.0)

    # : avg_strength, hit_spots, sum_strength
    g2 = g2.sort_values(["avg_strength", "hit_spots", "sum_strength"],
                         ascending=[False, False, False]).reset_index(drop=True)
    return g2

def topk_for_spot_inline(spot_idx: int, K: int = 20):
    """ single spot compute Top-K (in symbol),: - symbols_used: List[str] - results: List[List[(partner_symbol, score)]], and symbols_used align """
    # spot of gene and
    gids = ctx_genes[spot_idx]
    A_raw = A_list[spot_idx]

    # ID -> symbol
    symbols_raw = ids_to_clean_symbols(gids, id2sym)

    # align
    if A_raw.shape[0] != len(symbols_raw):
        minL = min(A_raw.shape[0], len(symbols_raw))
        if minL == 0:
            raise ValueError(f"spot {spot_idx} of or gene listis empty")
        A_raw = A_raw[:minL, :minL]
        symbols_raw = symbols_raw[:minL]

    # filter and
    symbols_filt, A_filt, kept_idx = filter_symbols_and_attention(
        symbols=symbols_raw, A=A_raw,
        drop_with_dot_or_dash=DROP_WITH_DOT_OR_DASH,
        extra_exclude_prefixes=EXTRA_EXCLUDE_PREFIXES,
        extra_exclude_regex=EXTRA_EXCLUDE_REGEX,
        whitelist=None
    )
    assert_no_forbidden_symbols(symbols_filt)

    # Top-K
    Asym = 0.5 * (A_filt + A_filt.T)
    results, symbols_used = topk_partners_per_symbol(Asym, symbols_filt, K=K, keep_self=False)
    assert_no_forbidden_symbols(symbols_used)
    return symbols_used, results

# ------------------ ------------------
def main():
    os.makedirs(result_output_dir, exist_ok=True)

    # PDAC
    # print("[INFO] Reading h5ad:", h5ad_path)
    h5ad_path = '/home/junning/projectnvme/ST/Data/PDAC/pdac.h5ad'
    truth_path =  '/home/junning/projectnvme/ST/Data/PDAC/PDAC_truth.csv'
    adata = sc.read_h5ad(h5ad_path)
    df_meta_layer = pd.read_csv(truth_path)["Region"]
    adata.obs['ground_truth'] = df_meta_layer.values




    print("Total spots (adata):", adata.n_obs)

    # vocab id2sym ()
    print("[INFO] Reading vocab:", vocab_path)
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    id2sym: Dict[int, str] = build_id2sym(vocab)

    # ctx_genes
    if not os.path.exists(ctx_genes_pkl_path):
        raise FileNotFoundError(f"Not found: {ctx_genes_pkl_path}")
    print("[INFO] Reading context_genes.pkl:", ctx_genes_pkl_path)
    with open(ctx_genes_pkl_path, "rb") as f:
        ctx_genes: List[np.ndarray] = pickle.load(f)
    print("context_genes.pkl -> list length:", len(ctx_genes))
    if len(ctx_genes) != adata.n_obs:
        print(f"[WARN] ctx_genes ({len(ctx_genes)}) and adata.n_obs({adata.n_obs})inconsistent. can in filter/.")

    # attention packs
    if not os.path.exists(attn_pkl_path):
        raise FileNotFoundError(f"Not found: {attn_pkl_path}")
    print("[INFO] Reading attention packs:", attn_pkl_path)
    with open(attn_pkl_path, "rb") as f:
        packs: List[Dict[str, Any]] = pickle.load(f)
    if not isinstance(packs, list) or len(packs) == 0:
        raise ValueError("attention packs formatinvalid,expected to benon-empty list")

    # attention: center
    layer_keys = sorted([k for k in packs[0].keys() if k.startswith("context_encoder_layer_")],
                        key=lambda s: int(s.rsplit("_", 1)[1]))
    layer_key = f"context_encoder_layer_{use_layer}"
    if layer_key not in packs[0]:
        raise KeyError(f"attention packs in no {layer_key},available: {layer_keys}")

    A_list: List[np.ndarray] = []
    total_centers = 0
    for b, p in enumerate(packs):
        A_raw = to_np(p[layer_key]["center_self_attention"])
        if A_raw.ndim == 3:
            N, L1, L2 = A_raw.shape
            if L1 != L2:
                raise ValueError(f"non-square matrix: {A_raw.shape} @ pack {b}")
            for n in range(N):
                A_list.append(A_raw[n])
            total_centers += N
        elif A_raw.ndim == 2:
            L1, L2 = A_raw.shape
            if L1 != L2:
                raise ValueError(f"non-square matrix: {A_raw.shape} @ pack {b}")
            A_list.append(A_raw)
            total_centers += 1
        else:
            raise ValueError(f"unexpected shape {A_raw.shape} @ pack {b}")

    print(f"Flattened centers = {len(A_list)} (sum N_center across packs = {total_centers})")
    if len(A_list) != len(ctx_genes):
        print(f"[WARN] attention ({len(A_list)}) and ctx_genes ({len(ctx_genes)})inconsistent.")
        print(" packs of ('spot_indices'/'obs_names'/'offset') or.")
        print(" before by align, can.")

    #  and  ctx_genes/adata align
    M = min(len(A_list), len(ctx_genes), adata.n_obs)
    if M == 0:
        raise ValueError("no available of  spot performanalysis (M==0)")
    if M < adata.n_obs:
        print(f"[INFO] trim:use the first M={M} spot performalignanalysis.")

    # to of
    globals().update({
        "adata": adata,
        "id2sym": id2sym,
        "ctx_genes": ctx_genes,
        "A_list": A_list,
        "M": M,
    })

    # check
    spot_names = np.array(adata.obs_names.tolist()[:M])
    if "ground_truth" not in adata.obs.columns:
        raise KeyError("adata.obs in in 'ground_truth' column.")
    clusters = np.array(adata.obs["ground_truth"].tolist()[:M])

    # Note.
    cluster_to_indices = defaultdict(list)
    for i in range(M):
        cluster_to_indices[clusters[i]].append(i)

    print(f"total cluster :{len(cluster_to_indices)}")

    # Output directory (each cluster CSV)
    per_cluster_out_dir = os.path.join(result_output_dir, f"hbrc_layer{use_layer}_clusters")
    os.makedirs(per_cluster_out_dir, exist_ok=True)

    # Top-K
    K = 20

    summary_rows = []
    cluster_dfs = {}  # each cluster of DataFrame, for after aggregation
    
    for clus, idx_list in cluster_to_indices.items():
        clus_str = str(clus)
        fname = f"cluster_{safe_filename(clus_str)}.csv"
        out_csv_path = os.path.join(per_cluster_out_dir, fname)

        # long tablerows
        long_rows = []
        for i in idx_list:
            try:
                symbols_used, results = topk_for_spot_inline(i, K=K)
                for gene, partners in zip(symbols_used, results):
                    for rank, (p_sym, score) in enumerate(partners[:K], start=1):
                        long_rows.append({
                            "spot_idx": i,
                            "spot_name": spot_names[i],
                            "cluster": clus_str,
                            "gene_symbol": gene,
                            "rank": rank,
                            "partner_symbol": p_sym,
                            "score": score,  # ,
                        })
                summary_rows.append({
                    "cluster": clus_str,
                    "spot_idx": i,
                    "spot_name": spot_names[i],
                    "cluster_csv_path": out_csv_path
                })
            except Exception as e:
                # error,
                print(f"[ERROR] spot {i} processingfailed: {e}")
                long_rows.append({
                    "spot_idx": i,
                    "spot_name": spot_names[i],
                    "cluster": clus_str,
                    "gene_symbol": "",
                    "rank": "",
                    "partner_symbol": "",
                    "score": f"ERROR: {e}",
                })

        # DataFrame
        df_cluster = pd.DataFrame(
            long_rows,
            columns=["spot_idx", "spot_name", "cluster", "gene_symbol", "rank", "partner_symbol", "score"]
        )
        # : by spot, by gene, by rank
        df_cluster.sort_values(by=["spot_idx", "gene_symbol", "rank"], inplace=True, na_position="last")
        
        # Savelong table (score for format)
        df_cluster_save = df_cluster.copy()
        df_cluster_save["score"] = df_cluster_save["score"].apply(lambda x: f"{x:.8g}" if isinstance(x, (int, float)) else x)
        df_cluster_save.to_csv(out_csv_path, index=False, encoding="utf-8")
        print(f"[OK] Save cluster CSV (long table):{out_csv_path} (rows: {len(long_rows)})")
        
        # for aggregation (score)
        cluster_dfs[clus_str] = df_cluster

    # Savetotalindex table
    summary_path = os.path.join(per_cluster_out_dir, "clusters_summary_index.csv")
    pd.DataFrame(summary_rows).sort_values(by=["cluster", "spot_idx"]).to_csv(summary_path, index=False, encoding="utf-8")
    print(f"[OK] Savetotalindex table:{summary_path}")

    print("[INFO] Startaggregationanalysis, enrichment prepared data...")
    # ============= of aggregation =============
    os.makedirs(enrichment_prepared_dir, exist_ok=True)
    dom_full_dir = os.path.join(enrichment_prepared_dir, "domain_tables")
    os.makedirs(dom_full_dir, exist_ok=True)

    dom_top_dict = {}
    all_rows = []

    for clus_str, df_long in cluster_dfs.items():
        # filtererrorrows (score of)
        if len(df_long) > 0:
            df_long = df_long[pd.to_numeric(df_long["score"], errors="coerce").notna()].copy()
            df_long["score"] = pd.to_numeric(df_long["score"])
        
        if len(df_long) == 0:
            print(f"[WARN] cluster {clus_str} valid data,Skippingaggregation")
            continue

        # aggregation
        agg = aggregate_domain_partners(df_long)

        # save domain table
        dom_full = agg.copy()
        dom_full.insert(0, "domain", clus_str)
        full_csv = os.path.join(dom_full_dir, f"{safe_filename(clus_str)}_partners_full.csv")
        dom_full.to_csv(full_csv, index=False)
        print(f"[SAVE] {full_csv}")

        # Top
        dom_top = agg.head(top_kv_per_domain).copy()
        dom_top.insert(0, "domain", clus_str)
        dom_top["rank"] = np.arange(1, len(dom_top) + 1)

        dom_top_dict[clus_str] = dom_top
        all_rows.append(dom_top)

    # all domain of Top partners
    if all_rows:
        all_top_df = pd.concat(all_rows, ignore_index=True)
    else:
        all_top_df = pd.DataFrame(columns=["domain", "partner_symbol", "hit_spots", "sum_strength", "avg_strength", "rank"])

    merged_csv = os.path.join(enrichment_prepared_dir, "all_domains_top_partners.csv")
    all_top_df.to_csv(merged_csv, index=False)
    print(f"[SAVE] {merged_csv}")

    print("[DONE] allCompleted.")
    print(f"  - long tableOutput directory:{per_cluster_out_dir}")
    print(f" - aggregationOutput directory:{enrichment_prepared_dir}")

# in when running the script directlyrows
if __name__ == "__main__":
    main()
