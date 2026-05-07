#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自注意力分析与富集准备一体化脚本

功能：
1. 导出每个 cluster 的长表格（spot-gene-partner 三元组）
2. 对每个 domain/cluster 进行聚合分析
3. 生成 Top partners 列表，为后续富集分析做准备

输出：
- result_output/hbrc_layer{N}_clusters/: 每个 cluster 的详细长表格 CSV
- result_output/enrichment_prepared/domain_tables/: 每个 domain 的完整聚合表
- result_output/enrichment_prepared/all_domains_top_partners.csv: 所有 domain 的 Top partners 汇总
"""

import os
import pickle
from typing import List, Dict, Optional, Sequence, Tuple, Any
import matplotlib
matplotlib.use("Agg")  # 无显示环境时安全
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

# ------------------ 基本路径（HBRC） ------------------
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

# 每个 domain 导出的 Top 基因数
top_kv_per_domain = 100

vocab_path = "/home/junning/projectnvme/ST/Data/Get_Embedding/vocab.json"
ctx_genes_pkl_path = os.path.join(base_dir, "context_genes.pkl")
attn_pkl_path = os.path.join(base_dir, "context_attention_scores.pkl")

# 选择分析层与样本
use_layer = 5        # 可改
# 单点调试索引用于测试函数，不会生成单点CSV
spot_idx_debug = 5

# ------------------ 过滤规则（按照基因过滤流程图）------------------
# 步骤1白名单：HLA-, MIR (在代码中硬编码)
# 步骤4前缀黑名单
EXTRA_EXCLUDE_PREFIXES = ("AC", "AL", "LINC", "RP", "SNOR", "SCARNA")
# 注意：MIR从黑名单移除，因为它在白名单中
DROP_WITH_DOT_OR_DASH = True  # 保留参数兼容性，但实际使用点号检查
EXTRA_EXCLUDE_REGEX = None  # 步骤5：自定义正则（可选）
DOT_DASH_PATTERN = re.compile(r"[.\-\u00B7\u2219\u2010\u2011\u2012\u2013\u2014\u2015]")

# ------------------ 工具函数 ------------------
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
    # 归一化 Unicode 横杠/中点到 ASCII
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
    # 优先认为 {symbol: id}
    id2sym_try1 = {}
    bad = 0
    for k, v in vd.items():
        try:
            id2sym_try1[int(v)] = str(k).strip()
        except Exception:
            bad += 1
    if bad > 0 and bad > len(vd) * 0.5:
        # 切换为 {id: symbol}
        id2sym_try2 = {}
        for k, v in vd.items():
            try:
                id2sym_try2[int(k)] = str(v).strip()
            except Exception:
                pass
        if not id2sym_try2:
            raise RuntimeError("无法从 vocab.json 构造 id2sym；请检查文件结构。")
        return id2sym_try2
    if not id2sym_try1:
        raise RuntimeError("vocab 映射为空；请检查文件结构。")
    return id2sym_try1

def filter_symbols_and_attention(
    symbols: Sequence[str],
    A: np.ndarray,
    drop_with_dot_or_dash: bool = True,
    extra_exclude_prefixes: Sequence[str] = EXTRA_EXCLUDE_PREFIXES,
    extra_exclude_regex: Optional[str] = EXTRA_EXCLUDE_REGEX,
    whitelist: Optional[set] = None
) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    过滤基因符号和注意力矩阵（按照基因过滤流程图）
    
    步骤1: 白名单检查 - 保留HLA-和MIR开头的基因
    步骤2: 点号检查 - 移除含点号的基因
    步骤3: 正则模式检查 - 排除特定模式的基因
    步骤4: 前缀黑名单检查 - 排除特定前缀的基因
    步骤5: 自定义正则检查 - 额外的用户自定义排除
    """
    L = len(symbols)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")
    # 对齐至 L
    if A.shape[0] != L:
        A = A[:L, :L]
    prefixes = tuple(p.lower() for p in (extra_exclude_prefixes or []))
    kept = []
    for i, s in enumerate(symbols):
        s_clean = clean_symbol(s)

        # 步骤1: 白名单检查 - 保留HLA-和MIR开头的基因
        if s_clean.startswith(('HLA-', 'MIR')):
            kept.append(i)
            continue

        # 步骤2: 点号检查 - 含点号的基因几乎都是低质量注释
        if "." in s_clean:
            continue

        # 步骤3: 正则模式检查 - 精确过滤假基因、反义RNA等
        regex_patterns = [
            r'^[A-Z]{2}\d+\.\d+$',      # AC012345.1 (基因组克隆)
            r'^ABBA\d+\.\d+$',           # ABBA01000934.1
            r'-AS\d*$',                  # A1BG-AS1 (反义RNA)
            r'^RP\d+-\w+\.\d+$',         # RP11-206L10.2 (假基因)
            r'^CTD-\w+\.\d+$',           # CTD假基因
            r'^CTC-\w+\.\d+$'            # CTC假基因
        ]
        if any(re.match(pattern, s_clean) for pattern in regex_patterns):
            continue

        # 步骤4: 前缀黑名单检查
        if prefixes and any(s_clean.lower().startswith(px) for px in prefixes):
            continue

        # 步骤5: 自定义正则检查（如果提供）
        if extra_exclude_regex:
            pat = re.compile(extra_exclude_regex)
            if pat.search(s_clean):
                continue

        kept.append(i)

    if len(kept) == 0:
        # 兜底：保留主对角最大
        diag = np.diag(A)
        kept = [int(np.nanargmax(diag))] if diag.size > 0 else [0]
    kept_idx = np.asarray(kept, dtype=int)
    A_filt = A[np.ix_(kept_idx, kept_idx)]
    symbols_filt = [clean_symbol(symbols[i]) for i in kept_idx.tolist()]
    return symbols_filt, A_filt, kept_idx.tolist()

def assert_no_forbidden_symbols(symbols: Sequence[str],
                                prefixes: Sequence[str] = EXTRA_EXCLUDE_PREFIXES,
                                drop_dot_dash: bool = DROP_WITH_DOT_OR_DASH):
    """
    验证过滤后的基因符号（排除白名单基因）
    """
    bad = []
    for s in symbols:
        s_clean = clean_symbol(s)
        
        # 跳过白名单基因的检查
        if s_clean.startswith(('HLA-', 'MIR')):
            continue
            
        # 检查点号
        if "." in s_clean:
            bad.append(("dot", s_clean, repr(s)))
        
        # 检查前缀黑名单
        if any(s_clean.lower().startswith(px.lower()) for px in prefixes):
            bad.append(("prefix", s_clean, repr(s)))
    
    if bad:
        print("[ERROR] 过滤后仍检测到非法符号（示例最多显示前 30 个）：")
        for kind, s_clean, r in bad[:30]:
            print(" -", kind, ":", s_clean, "raw_repr:", r)
        raise AssertionError("过滤失败：仍存在非法符号。")

def topk_partners_per_symbol(A_sym: np.ndarray, symbols: Sequence[str], K: int = 20, keep_self: bool = False):
    L = len(symbols)
    if A_sym.ndim != 2 or A_sym.shape[0] < L or A_sym.shape[1] < L:
        raise ValueError(f"A_sym 尺寸不匹配: {A_sym.shape} vs L={L}")
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

# ------------------ 按 cluster 的导出（长表格）与聚合函数 ------------------
def safe_filename(s: str, repl: str = "_") -> str:
    """将任意字符串清洗为可用于文件名的安全形式。"""
    if s is None:
        return "NA"
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = re.sub(r"[\\/:\*\?\"<>\|\s]+", repl, s)
    s = s.strip(repl)
    return s or "NA"

def aggregate_domain_partners(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    对单个 domain 的长表执行两级聚合：
      - 级别1（spot 内）：partner_symbol 在每个 spot 的 score_sum_by_spot = sum(score)
      - 级别2（domain 内）：对 partner_symbol 的 score_sum_by_spot 求均值与命中数
    返回列：
      partner_symbol, hit_spots, sum_strength, avg_strength
    其中：
      hit_spots = 出现过的不同 spot 数
      sum_strength = 所有出现 spot 的 score_sum_by_spot 之和
      avg_strength = sum_strength / hit_spots
    """
    # 级别1：spot 内汇总
    g1 = (df_long.groupby(["spot_idx", "partner_symbol"], as_index=False)["score"]
                 .sum()
                 .rename(columns={"score": "score_sum_by_spot"}))

    # 级别2：domain 内聚合
    g2 = (g1.groupby("partner_symbol", as_index=False)
             .agg(hit_spots=("spot_idx", "nunique"),
                  sum_strength=("score_sum_by_spot", "sum")))

    g2["avg_strength"] = g2["sum_strength"] / g2["hit_spots"].replace(0, np.nan)
    g2["avg_strength"] = g2["avg_strength"].fillna(0.0)

    # 排序建议：优先 avg_strength，其次 hit_spots，再次 sum_strength
    g2 = g2.sort_values(["avg_strength", "hit_spots", "sum_strength"],
                         ascending=[False, False, False]).reset_index(drop=True)
    return g2

def topk_for_spot_inline(spot_idx: int, K: int = 20):
    """
    对单个 spot 现场计算 Top-K（在 symbol 空间），返回：
    - symbols_used: List[str]
    - results: List[List[(partner_symbol, score)]], 与 symbols_used 对齐
    """
    # 取该 spot 的基因与注意力矩阵
    gids = ctx_genes[spot_idx]
    A_raw = A_list[spot_idx]

    # 映射 ID -> symbol 并清洗
    symbols_raw = ids_to_clean_symbols(gids, id2sym)

    # 尺寸对齐
    if A_raw.shape[0] != len(symbols_raw):
        minL = min(A_raw.shape[0], len(symbols_raw))
        if minL == 0:
            raise ValueError(f"spot {spot_idx} 的注意力矩阵或基因列表为空")
        A_raw = A_raw[:minL, :minL]
        symbols_raw = symbols_raw[:minL]

    # 过滤与审计
    symbols_filt, A_filt, kept_idx = filter_symbols_and_attention(
        symbols=symbols_raw, A=A_raw,
        drop_with_dot_or_dash=DROP_WITH_DOT_OR_DASH,
        extra_exclude_prefixes=EXTRA_EXCLUDE_PREFIXES,
        extra_exclude_regex=EXTRA_EXCLUDE_REGEX,
        whitelist=None
    )
    assert_no_forbidden_symbols(symbols_filt)

    # 对称化并 Top-K
    Asym = 0.5 * (A_filt + A_filt.T)
    results, symbols_used = topk_partners_per_symbol(Asym, symbols_filt, K=K, keep_self=False)
    assert_no_forbidden_symbols(symbols_used)
    return symbols_used, results

# ------------------ 主流程 ------------------
def main():
    os.makedirs(result_output_dir, exist_ok=True)

    # 读数据  PDAC
    # print("[INFO] 读取 h5ad:", h5ad_path)
    h5ad_path = '/home/junning/projectnvme/ST/Data/PDAC/pdac.h5ad'
    truth_path =  '/home/junning/projectnvme/ST/Data/PDAC/PDAC_truth.csv'
    adata = sc.read_h5ad(h5ad_path)
    df_meta_layer = pd.read_csv(truth_path)["Region"]
    adata.obs['ground_truth'] = df_meta_layer.values




    print("Total spots (adata):", adata.n_obs)

    # 读 vocab 并构造 id2sym（自动判断方向）
    print("[INFO] 读取 vocab:", vocab_path)
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    id2sym: Dict[int, str] = build_id2sym(vocab)

    # 读 ctx_genes
    if not os.path.exists(ctx_genes_pkl_path):
        raise FileNotFoundError(f"Not found: {ctx_genes_pkl_path}")
    print("[INFO] 读取 context_genes.pkl:", ctx_genes_pkl_path)
    with open(ctx_genes_pkl_path, "rb") as f:
        ctx_genes: List[np.ndarray] = pickle.load(f)
    print("context_genes.pkl -> list length:", len(ctx_genes))
    if len(ctx_genes) != adata.n_obs:
        print(f"[WARN] ctx_genes 数({len(ctx_genes)})与 adata.n_obs({adata.n_obs})不一致。可能存在过滤/顺序差异。")

    # 读 attention packs
    if not os.path.exists(attn_pkl_path):
        raise FileNotFoundError(f"Not found: {attn_pkl_path}")
    print("[INFO] 读取 attention packs:", attn_pkl_path)
    with open(attn_pkl_path, "rb") as f:
        packs: List[Dict[str, Any]] = pickle.load(f)
    if not isinstance(packs, list) or len(packs) == 0:
        raise ValueError("attention packs 格式异常，期望为非空 list")

    # 展平 attention：逐 center 展平
    layer_keys = sorted([k for k in packs[0].keys() if k.startswith("context_encoder_layer_")],
                        key=lambda s: int(s.rsplit("_", 1)[1]))
    layer_key = f"context_encoder_layer_{use_layer}"
    if layer_key not in packs[0]:
        raise KeyError(f"attention packs 中没有 {layer_key}，可用键: {layer_keys}")

    A_list: List[np.ndarray] = []
    total_centers = 0
    for b, p in enumerate(packs):
        A_raw = to_np(p[layer_key]["center_self_attention"])
        if A_raw.ndim == 3:
            N, L1, L2 = A_raw.shape
            if L1 != L2:
                raise ValueError(f"非方阵: {A_raw.shape} @ pack {b}")
            for n in range(N):
                A_list.append(A_raw[n])
            total_centers += N
        elif A_raw.ndim == 2:
            L1, L2 = A_raw.shape
            if L1 != L2:
                raise ValueError(f"非方阵: {A_raw.shape} @ pack {b}")
            A_list.append(A_raw)
            total_centers += 1
        else:
            raise ValueError(f"unexpected shape {A_raw.shape} @ pack {b}")

    print(f"Flattened centers = {len(A_list)} (sum N_center across packs = {total_centers})")
    if len(A_list) != len(ctx_genes):
        print(f"[WARN] attention 数({len(A_list)})与 ctx_genes 数({len(ctx_genes)})不一致。")
        print("      需要 packs 内的全局索引键（例如 'spot_indices'/'obs_names'/'offset' 等）来重排或补齐。")
        print("      当前将按最小长度对齐，可能丢弃末尾元素。")

    # 与 ctx_genes/adata 对齐
    M = min(len(A_list), len(ctx_genes), adata.n_obs)
    if M == 0:
        raise ValueError("没有可用的 spot 进行分析（M==0）")
    if M < adata.n_obs:
        print(f"[INFO] 有裁剪：将使用前 M={M} 个 spot 进行对齐分析。")

    # 暴露到外层函数需要的闭包变量
    globals().update({
        "adata": adata,
        "id2sym": id2sym,
        "ctx_genes": ctx_genes,
        "A_list": A_list,
        "M": M,
    })

    # 基本检查
    spot_names = np.array(adata.obs_names.tolist()[:M])
    if "ground_truth" not in adata.obs.columns:
        raise KeyError("adata.obs 中不存在 'ground_truth' 列。")
    clusters = np.array(adata.obs["ground_truth"].tolist()[:M])

    # 分组
    cluster_to_indices = defaultdict(list)
    for i in range(M):
        cluster_to_indices[clusters[i]].append(i)

    print(f"总 cluster 数：{len(cluster_to_indices)}")

    # 输出目录（每个 cluster 一个 CSV）
    per_cluster_out_dir = os.path.join(result_output_dir, f"hbrc_layer{use_layer}_clusters")
    os.makedirs(per_cluster_out_dir, exist_ok=True)

    # Top-K
    K = 20

    summary_rows = []
    cluster_dfs = {}  # 存储每个 cluster 的 DataFrame，用于后续聚合
    
    for clus, idx_list in cluster_to_indices.items():
        clus_str = str(clus)
        fname = f"cluster_{safe_filename(clus_str)}.csv"
        out_csv_path = os.path.join(per_cluster_out_dir, fname)

        # 长表格行
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
                            "score": score,  # 保持浮点数，不转字符串
                        })
                summary_rows.append({
                    "cluster": clus_str,
                    "spot_idx": i,
                    "spot_name": spot_names[i],
                    "cluster_csv_path": out_csv_path
                })
            except Exception as e:
                # 记录错误，便于定位问题
                print(f"[ERROR] spot {i} 处理失败: {e}")
                long_rows.append({
                    "spot_idx": i,
                    "spot_name": spot_names[i],
                    "cluster": clus_str,
                    "gene_symbol": "",
                    "rank": "",
                    "partner_symbol": "",
                    "score": f"ERROR: {e}",
                })

        # DataFrame 并写出
        df_cluster = pd.DataFrame(
            long_rows,
            columns=["spot_idx", "spot_name", "cluster", "gene_symbol", "rank", "partner_symbol", "score"]
        )
        # 排序：先按 spot，再按 gene，再按 rank
        df_cluster.sort_values(by=["spot_idx", "gene_symbol", "rank"], inplace=True, na_position="last")
        
        # 保存长表格（score 转为字符串格式）
        df_cluster_save = df_cluster.copy()
        df_cluster_save["score"] = df_cluster_save["score"].apply(lambda x: f"{x:.8g}" if isinstance(x, (int, float)) else x)
        df_cluster_save.to_csv(out_csv_path, index=False, encoding="utf-8")
        print(f"[OK] 保存 cluster CSV（长表格）：{out_csv_path} (rows: {len(long_rows)})")
        
        # 存储用于聚合（保持数值型 score）
        cluster_dfs[clus_str] = df_cluster

    # 保存总索引表
    summary_path = os.path.join(per_cluster_out_dir, "clusters_summary_index.csv")
    pd.DataFrame(summary_rows).sort_values(by=["cluster", "spot_idx"]).to_csv(summary_path, index=False, encoding="utf-8")
    print(f"[OK] 保存总索引表：{summary_path}")

    print("[INFO] 开始聚合分析，生成 enrichment 准备数据...")
    # ============= 融合的聚合功能 =============
    os.makedirs(enrichment_prepared_dir, exist_ok=True)
    dom_full_dir = os.path.join(enrichment_prepared_dir, "domain_tables")
    os.makedirs(dom_full_dir, exist_ok=True)

    dom_top_dict = {}
    all_rows = []

    for clus_str, df_long in cluster_dfs.items():
        # 过滤掉错误行（score 不是数值的）
        if len(df_long) > 0:
            df_long = df_long[pd.to_numeric(df_long["score"], errors="coerce").notna()].copy()
            df_long["score"] = pd.to_numeric(df_long["score"])
        
        if len(df_long) == 0:
            print(f"[WARN] cluster {clus_str} 无有效数据，跳过聚合")
            continue

        # 聚合
        agg = aggregate_domain_partners(df_long)

        # 保存完整 domain 表
        dom_full = agg.copy()
        dom_full.insert(0, "domain", clus_str)
        full_csv = os.path.join(dom_full_dir, f"{safe_filename(clus_str)}_partners_full.csv")
        dom_full.to_csv(full_csv, index=False)
        print(f"[SAVE] {full_csv}")

        # 取 Top
        dom_top = agg.head(top_kv_per_domain).copy()
        dom_top.insert(0, "domain", clus_str)
        dom_top["rank"] = np.arange(1, len(dom_top) + 1)

        dom_top_dict[clus_str] = dom_top
        all_rows.append(dom_top)

    # 合并所有 domain 的 Top partners
    if all_rows:
        all_top_df = pd.concat(all_rows, ignore_index=True)
    else:
        all_top_df = pd.DataFrame(columns=["domain", "partner_symbol", "hit_spots", "sum_strength", "avg_strength", "rank"])

    merged_csv = os.path.join(enrichment_prepared_dir, "all_domains_top_partners.csv")
    all_top_df.to_csv(merged_csv, index=False)
    print(f"[SAVE] {merged_csv}")

    print("[DONE] 全部完成。")
    print(f"  - 长表格输出目录：{per_cluster_out_dir}")
    print(f"  - 聚合结果输出目录：{enrichment_prepared_dir}")

# 仅在直接运行脚本时执行
if __name__ == "__main__":
    main()