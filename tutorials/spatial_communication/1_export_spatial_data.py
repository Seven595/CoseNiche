"""
增强版空间注意力数据导出脚本
- 优化处理全切片空间转录组数据
- 导出多种优化数据结构用于高效分析和可视化
- 添加详细进度显示
- 添加无Parquet依赖库时的回退选项
- 改进的基因过滤策略：保留HLA等重要基因，精确过滤低质量注释

更新说明 (2025-10-29):
- 改进基因过滤策略，不再粗暴移除所有含横杠的基因
- 使用白名单保留重要基因（HLA-*, MIR*等）
- 使用正则模式精确过滤假基因、基因组克隆、反义RNA
- 保留约60.6%的基因（vs旧策略59.9%），包括所有38个HLA基因
"""

import os
import pickle
import time
from typing import List, Dict, Optional, Sequence, Tuple, Any
import scanpy as sc
import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict
import warnings
import argparse
from datetime import datetime
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("警告: 未找到tqdm库，将使用简化进度显示")


# 忽略不必要的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 基因过滤设置（按照基因过滤流程图）
EXTRA_EXCLUDE_PREFIXES = ("AC", "AL", "SNOR", "SCARNA", "LINC")
DROP_WITH_DOT_OR_DASH = False  # 改进策略：不再粗暴过滤所有横杠
EXTRA_EXCLUDE_REGEX = None

# ------------------ 进度显示函数 ------------------
def get_time_estimate(start_time, current_step, total_steps):
    """计算估计剩余时间"""
    elapsed = time.time() - start_time
    if current_step == 0:
        return "计算中..."
    
    steps_per_second = current_step / elapsed
    remaining_seconds = (total_steps - current_step) / steps_per_second
    
    # 转换为可读格式
    if remaining_seconds < 60:
        return f"{remaining_seconds:.1f}秒"
    elif remaining_seconds < 3600:
        return f"{remaining_seconds/60:.1f}分钟"
    else:
        return f"{remaining_seconds/3600:.1f}小时"

def create_progress_bar(total, desc="处理中"):
    """创建进度条"""
    if TQDM_AVAILABLE:
        return tqdm(total=total, desc=desc, unit="spot", ncols=100, 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    else:
        # 简单的进度显示类
        class SimpleProgressBar:
            def __init__(self, total, desc):
                self.total = total
                self.desc = desc
                self.n = 0
                self.start_time = time.time()
                
            def update(self, n=1):
                self.n += n
                elapsed = time.time() - self.start_time
                if self.n % 10 == 0 or self.n == self.total:
                    print(f"{self.desc}: {self.n}/{self.total} [{elapsed:.1f}s 已用]")
                    
            def set_postfix(self, **kwargs):
                # 简单版本不显示postfix
                pass
                
            def close(self):
                elapsed = time.time() - self.start_time
                print(f"{self.desc}: 完成 {self.n}/{self.total} [{elapsed:.1f}s 总用时]")
                
        return SimpleProgressBar(total, desc)

def log_progress(message, log_file=None):
    """记录进度消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # 如果提供了日志文件，则写入日志
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')

# ------------------ 数据保存与加载函数 ------------------
def save_dataframe(df, path, format_hint="auto"):
    """保存DataFrame，自动处理不同格式"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # 确定文件格式
    if format_hint == "auto":
        if path.endswith(".parquet"):
            format_hint = "parquet"
        elif path.endswith(".csv"):
            format_hint = "csv"
        else:
            # 默认使用CSV
            format_hint = "csv"
            if not path.endswith(".csv"):
                path = path + ".csv"
    
    # 根据指定格式保存
    df.to_csv(path, index=False)
    return path

def load_dataframe(path):
    """加载DataFrame，自动处理不同格式"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    
    return pd.read_csv(path)

# ------------------ 工具函数 ------------------
def to_np(x):
    """转换为NumPy数组"""
    if isinstance(x, np.ndarray): return x
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.asarray(x)

def clean_symbol(s: str) -> str:
    """清理基因符号"""
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

DOT_DASH_PATTERN = re.compile(r"[.\-\u00B7\u2219\u2010\u2011\u2012\u2013\u2014\u2015]")

def build_id2sym(vd: Dict[Any, Any]) -> Dict[int, str]:
    """构建ID到符号的映射"""
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

def ids_to_clean_symbols(gids: Sequence[Any], id2sym: Dict[int, str]) -> List[str]:
    """将ID列表转换为清洁的符号列表"""
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

def is_coding_gene(symbol: str, 
                  drop_with_dot_or_dash: bool = DROP_WITH_DOT_OR_DASH,
                  extra_exclude_prefixes: Sequence[str] = EXTRA_EXCLUDE_PREFIXES,
                  extra_exclude_regex: Optional[str] = EXTRA_EXCLUDE_REGEX) -> bool:
    """
    判断一个基因符号是否是编码基因（改进版）
    
    按照基因过滤流程图：
    步骤1: 白名单检查 - 保留HLA-和MIR开头的基因
    步骤2: 点号检查 - 移除含点号的基因
    步骤3: 正则模式检查 - 排除特定模式的基因
    步骤4: 前缀黑名单检查 - 排除特定前缀的基因
    步骤5: 自定义正则检查 - 额外的用户自定义排除
    """
    symbol = clean_symbol(symbol)
    if not symbol:
        return False
    
    symbol_lower = symbol.lower()
    
    # 步骤1: 白名单检查 - 保留重要基因前缀（HLA-, MIR）
    if symbol.startswith(('HLA-', 'MIR')):
        return True
    
    # 步骤2: 点号检查 - 含点号的基因几乎都是低质量注释
    if "." in symbol:
        return False
    
    # 步骤3: 正则模式检查 - 精确过滤假基因、反义RNA等
    regex_patterns = [
        r'^[A-Z]{2}\d+\.\d+$',      # AC012345.1 (基因组克隆)
        r'^ABBA\d+\.\d+$',           # ABBA01000934.1
        r'-AS\d*$',                  # A1BG-AS1 (反义RNA)
        r'^RP\d+-\w+\.\d+$',         # RP11-206L10.2 (假基因)
        r'^CTD-\w+\.\d+$',           # CTD假基因
        r'^CTC-\w+\.\d+$'            # CTC假基因
    ]
    for pattern in regex_patterns:
        if re.match(pattern, symbol):
            return False
    
    # 步骤4: 前缀黑名单检查（AC, AL, SNOR, SCARNA, LINC等）
    prefixes = tuple(p.lower() for p in (extra_exclude_prefixes or []))
    if prefixes and any(symbol_lower.startswith(px) for px in prefixes):
        return False
    
    # 步骤5: 自定义正则检查（如果提供）
    if extra_exclude_regex:
        pat = re.compile(extra_exclude_regex)
        if pat.search(symbol):
            return False
    
    return True

def filter_genes_by_symbol(gene_ids: List[int], 
                          id2sym: Dict[int, str],
                          drop_with_dot_or_dash: bool = DROP_WITH_DOT_OR_DASH,
                          extra_exclude_prefixes: Sequence[str] = EXTRA_EXCLUDE_PREFIXES,
                          extra_exclude_regex: Optional[str] = EXTRA_EXCLUDE_REGEX,
                          verbose: bool = True) -> Tuple[List[int], List[int], List[str]]:
    """过滤基因ID列表，仅保留编码基因"""
    valid_indices = []
    filtered_ids = []
    filtered_symbols = []
    
    # 生成原始符号列表
    original_symbols = []
    for gid in gene_ids:
        try:
            sym = id2sym.get(int(gid), str(gid))
            original_symbols.append(sym)
        except Exception:
            original_symbols.append(str(gid))
    
    # 输出调试信息
    if verbose:
        log_progress(f"开始过滤，原始基因数量: {len(gene_ids)}")
        log_progress(f"前5个基因ID及其符号:")
        for i in range(min(5, len(gene_ids))):
            gid = gene_ids[i]
            sym = original_symbols[i]
            is_filtered = not is_coding_gene(sym, drop_with_dot_or_dash, extra_exclude_prefixes, extra_exclude_regex)
            log_progress(f"  [{i}] ID={gid}, Symbol={sym}, 将被过滤: {is_filtered}")
        
        # 统计原始数据中的非编码基因数量
        non_coding_count = sum(1 for sym in original_symbols if not is_coding_gene(sym, drop_with_dot_or_dash, extra_exclude_prefixes, extra_exclude_regex))
        log_progress(f"原始数据中非编码基因: {non_coding_count}/{len(original_symbols)} ({non_coding_count/len(original_symbols)*100:.1f}%)")
    
    # 执行过滤
    for i, gid in enumerate(gene_ids):
        try:
            sym = original_symbols[i]
            # 如果是编码基因，则保留
            if is_coding_gene(sym, drop_with_dot_or_dash, extra_exclude_prefixes, extra_exclude_regex):
                valid_indices.append(i)
                filtered_ids.append(gid)
                filtered_symbols.append(clean_symbol(sym))
        except Exception as e:
            if verbose:
                log_progress(f"警告: 处理基因ID {gid} 时出错: {e}")
    
    # 输出过滤结果
    if verbose:
        log_progress(f"过滤完成，保留基因数量: {len(filtered_ids)}/{len(gene_ids)} ({len(filtered_ids)/len(gene_ids)*100:.1f}%)")
        if filtered_symbols:
            log_progress(f"过滤后前5个基因符号: {filtered_symbols[:5]}")
    
    return filtered_ids, valid_indices, filtered_symbols

# ------------------ 空间注意力数据处理函数 ------------------
def flatten_packs_to_centers(
    packs: List[Dict[str, Any]],
    layer_key: str,
    global_idx_to_name: Optional[Dict[int,str]] = None,
    log_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    从 packs[layer_key]['spatial_cross_attention'] 打平为 per-center 条目。
    期望 weights 形状为 [C, Q, K] 或 [C, H, Q, K]（自动对头平均）。
    """
    log_progress(f"打平attention packs数据（层: {layer_key}）...", log_file)
    flat = []
    total_packs = len(packs)
    
    for pi, pack in enumerate(packs):
        if (pi + 1) % 10 == 0 or pi == 0 or pi == total_packs - 1:
            log_progress(f"处理pack {pi+1}/{total_packs}...", log_file)
        
        if layer_key not in pack:
            continue
        lp = pack[layer_key]
        if "spatial_cross_attention" not in lp:
            continue
        sca = lp["spatial_cross_attention"]

        W = to_np(sca.get("weights"))                  # [C,Q,K] or [C,H,Q,K]
        slices_per_center = sca.get("kv_source_slices", [])  # List[List[dict]] or List[dict]
        q_valid = to_np(sca.get("q_valid_lengths"))    # [C]
        kv_gene_ids = to_np(sca.get("kv_gene_ids"))    # [C,K] or None
        query_gene_ids = to_np(sca.get("query_gene_ids"))  # [C,Q] or None
        btlg = to_np(sca.get("batch_local_to_global"))     # [num_spots_in_pack] or None
        centers_global = to_np(sca.get("centers_global_indices"))  # [C]

        if W is None or q_valid is None or centers_global is None:
            continue

        # 平均多头
        if W.ndim == 4:
            # [C, H, Q, K] -> mean over H
            W = W.mean(axis=1)
        elif W.ndim != 3:
            raise ValueError(f"Unexpected weights shape: {W.shape}")

        C, q_max_in_W, Kmax = W.shape

        # 对齐 slices
        if isinstance(slices_per_center, list):
            if len(slices_per_center) == C and (len(slices_per_center) == 0 or isinstance(slices_per_center[0], list)):
                slices_aligned = slices_per_center
            elif len(slices_per_center) > 0 and isinstance(slices_per_center[0], dict):
                # 单一列表，广播到每个中心
                slices_aligned = [slices_per_center for _ in range(C)]
            else:
                raise ValueError("kv_source_slices must be List[List[dict]] or List[dict].")
        else:
            raise ValueError("kv_source_slices must be a list.")

        for i in range(C):
            center_global = int(centers_global[i])
            center_name = global_idx_to_name.get(center_global, "") if global_idx_to_name is not None else ""

            entry = dict(
                pack_index=int(pi),
                i_center_in_pack=int(i),
                center_global_idx=center_global,
                center_name=center_name,
                weights=W[i, :, :].astype(np.float32),
                kv_source_slices=slices_aligned[i],
                q_valid_len=int(q_valid[i]),
                neighbor_max_len=int(Kmax),
                q_max=int(q_max_in_W),
                batch_local_to_global=btlg,
                kv_gene_ids=(kv_gene_ids[i, :] if kv_gene_ids is not None else None),
                query_gene_ids=(query_gene_ids[i, :int(q_valid[i])] if query_gene_ids is not None else None),
            )
            flat.append(entry)
    
    log_progress(f"已打平为{len(flat)}个中心spots", log_file)
    return flat

def neighbor_groups_for_center(slices_i: List[Dict[str, int]]) -> List[Tuple[int, List[Tuple[int,int]]]]:
    """获取中心的邻居分组"""
    segs = defaultdict(list)
    for d in slices_i:
        nl = int(d.get("spot_idx", d.get("neighbor_local_row")))
        s = int(d["start"]); e = int(d["end"])
        if s < e:
            segs[nl].append((s, e))
    return [(nl, sorted(v)) for nl, v in segs.items()]

def top_k_keys_in_neighbor_for_query(
    W_i: np.ndarray,           # [q_max, Kmax]
    segs: List[Tuple[int,int]],
    q_index: int,
    topk: int
) -> Tuple[np.ndarray, np.ndarray]:
    """获取邻居中的Top-K键"""
    Kmax = W_i.shape[1]
    cols = []
    for s, e in segs:
        e = min(e, Kmax)
        if 0 <= s < e:
            cols.extend(range(s, e))
    if not cols:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    cols = np.asarray(cols, dtype=int)
    scores = W_i[q_index, cols]
    
    # 安全检查：确保topk不大于实际可用的键数量
    actual_topk = min(topk, len(cols))
    
    order = np.argsort(-scores)[:actual_topk]
    return cols[order], scores[order]

def process_all_spots_efficiently(packs, layer_key, adata, id2sym, global_idx_to_name, output_dir, log_file, batch_size=50):
    """高效处理和存储整个切片的所有spots"""
    log_progress(f"开始处理整个切片的所有spots（层：{layer_key}）...", log_file)
    
    # 打平所有中心spots的注意力数据
    flat_centers = flatten_packs_to_centers(
        packs=packs,
        layer_key=layer_key,
        global_idx_to_name=global_idx_to_name,
        log_file=log_file
    )
    
    total_centers = len(flat_centers)
    log_progress(f"共找到{total_centers}个中心spots", log_file)
    
    # 创建用于保存全部数据的大型DataFrame
    all_rows = []
    # batch_size 已作为参数传入
    
    # 分批处理以节省内存
    total_batches = (total_centers - 1) // batch_size + 1
    start_time = time.time()
    
    for batch_idx in range(0, total_centers, batch_size):
        batch_num = batch_idx // batch_size + 1
        log_progress(f"处理batch {batch_num}/{total_batches}...", log_file)
        
        # 估计剩余时间
        if batch_idx > 0:
            elapsed = time.time() - start_time
            spots_per_second = batch_idx / elapsed
            remaining_spots = total_centers - batch_idx
            remaining_seconds = remaining_spots / spots_per_second
            
            if remaining_seconds < 60:
                time_str = f"{remaining_seconds:.1f}秒"
            elif remaining_seconds < 3600:
                time_str = f"{remaining_seconds/60:.1f}分钟"
            else:
                time_str = f"{remaining_seconds/3600:.1f}小时"
                
            log_progress(f"  预计剩余时间: {time_str}", log_file)
        
        end_idx = min(batch_idx + batch_size, total_centers)
        batch_centers = flat_centers[batch_idx:end_idx]
        
        # 为当前batch创建进度条
        pbar = create_progress_bar(len(batch_centers), desc=f"Batch {batch_num}/{total_batches}")
        
        # 处理每个batch中的spot
        for i, center_entry in enumerate(batch_centers):
            current_idx = batch_idx + i
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(spot=current_idx+1, total=total_centers)
            
            # 获取该spot的注意力信息
            W_i = center_entry["weights"]              # [q_max,Kmax]
            qlen = int(center_entry["q_valid_len"])
            slices_i = center_entry["kv_source_slices"]
            btlg = center_entry.get("batch_local_to_global", None)
            kv_gene_ids = center_entry.get("kv_gene_ids", None)
            query_gene_ids_arr = center_entry.get("query_gene_ids", None)
            center_global = int(center_entry["center_global_idx"])
            center_name = center_entry.get("center_name", "")
            Kmax = W_i.shape[1]

            # 处理query基因
            filtered_q_indices = []
            filtered_q_genes = []
            
            if query_gene_ids_arr is not None:
                query_genes = []
                for q in range(qlen):
                    if q < len(query_gene_ids_arr):
                        maybe_gid = int(query_gene_ids_arr[q])
                        if maybe_gid >= 0:
                            query_genes.append(maybe_gid)
                        else:
                            query_genes.append(None)
                    else:
                        query_genes.append(None)
                
                # 过滤query基因
                for q, gid in enumerate(query_genes):
                    if gid is None:
                        continue
                        
                    if id2sym is not None:
                        sym = id2sym.get(gid, "")
                        # 检查是否是编码基因
                        if is_coding_gene(sym):
                            filtered_q_indices.append(q)
                            filtered_q_genes.append(gid)
                    else:
                        filtered_q_indices.append(q)
                        filtered_q_genes.append(gid)
            else:
                # 无query基因信息，使用全部
                filtered_q_indices = list(range(qlen))
                filtered_q_genes = [None] * qlen
            
            # 邻居分组
            groups = neighbor_groups_for_center(slices_i)
            
            # 处理每个query基因与其邻居
            for q_idx, q in enumerate(filtered_q_indices):
                # 获取query基因信息
                q_gid = filtered_q_genes[q_idx] if filtered_q_genes[q_idx] is not None else ""
                q_sym = ""
                if q_gid and id2sym is not None:
                    q_sym = id2sym.get(q_gid, "")

                # 计算每个邻居的注意力总和
                neighbor_items = []
                total_sum = 0.0
                
                for nl, segs in groups:
                    ssum = 0.0
                    for s, e in segs:
                        e = min(e, Kmax)
                        if 0 <= s < e:
                            ssum += float(W_i[q, s:e].sum())
                    
                    nei_global = int(btlg[nl]) if (btlg is not None) else None
                    nei_name = global_idx_to_name.get(nei_global, "") if (nei_global is not None and global_idx_to_name is not None) else ""
                    
                    if ssum > 0:  # 只存储有效注意力的邻居
                        neighbor_items.append(dict(
                            neighbor_local_row=int(nl),
                            neighbor_global_idx=nei_global,
                            neighbor_name=nei_name,
                            attn_sum=float(ssum)
                        ))
                        total_sum += ssum

                # 归一化邻居注意力
                for it in neighbor_items:
                    if total_sum > 0:
                        it["attn_sum_norm"] = it["attn_sum"] / total_sum
                    else:
                        it["attn_sum_norm"] = 0.0

                # 排序并只保留top邻居
                if neighbor_items:
                    neighbor_items.sort(key=lambda x: x["attn_sum"], reverse=True)
                    top_neighbors = neighbor_items[:7]  # 只保留前7个邻居
                    
                    # 对每个邻居提取top键基因
                    for it in top_neighbors:
                        nl = it["neighbor_local_row"]
                        segs = next((s for n, s in groups if n == nl), [])
                        
                        # 获取邻居内部的top键
                        cols, scores = top_k_keys_in_neighbor_for_query(W_i, segs, q_index=q, topk=10)
                        
                        if cols.size > 0:
                            # 处理该邻居内部的键基因
                            for c_abs, sc in zip(cols, scores):
                                kv_gid = ""
                                kv_sym = ""
                                
                                if kv_gene_ids is not None:
                                    gid = int(kv_gene_ids[int(c_abs)])
                                    if gid >= 0:
                                        kv_gid = gid
                                        if id2sym is not None:
                                            kv_sym = id2sym.get(kv_gid, "")
                                
                                # 添加一行完整的数据
                                row = {
                                    'layer_key': layer_key,
                                    'center_global_idx': center_global,
                                    'center_name': center_name,
                                    'q_index': int(q),
                                    'q_gene_id': q_gid,
                                    'q_gene_symbol': q_sym,
                                    'neighbor_global_idx': it["neighbor_global_idx"],
                                    'neighbor_name': it["neighbor_name"],
                                    'attn_sum': it["attn_sum"],
                                    'attn_sum_norm': it["attn_sum_norm"],
                                    'kv_abs_col': int(c_abs),
                                    'attn_score': float(sc),
                                    'kv_gene_id': kv_gid,
                                    'kv_gene_symbol': kv_sym
                                }
                                all_rows.append(row)
                        else:
                            # 至少添加一行记录邻居级别的注意力
                            row = {
                                'layer_key': layer_key,
                                'center_global_idx': center_global,
                                'center_name': center_name,
                                'q_index': int(q),
                                'q_gene_id': q_gid,
                                'q_gene_symbol': q_sym,
                                'neighbor_global_idx': it["neighbor_global_idx"],
                                'neighbor_name': it["neighbor_name"],
                                'attn_sum': it["attn_sum"],
                                'attn_sum_norm': it["attn_sum_norm"],
                                'kv_abs_col': "",
                                'attn_score': "",
                                'kv_gene_id': "",
                                'kv_gene_symbol': ""
                            }
                            all_rows.append(row)
        
        # 关闭进度条
        pbar.close()
        
        # 每处理完一批就保存一次中间结果
        if all_rows:
            # 保存批次数据
            batch_df = pd.DataFrame(all_rows)
            batch_path = os.path.join(output_dir, f"attention_batch_{batch_idx//batch_size}")
            
            # 使用通用保存函数，自动处理格式    
            saved_path = save_dataframe(batch_df, batch_path + ".csv")
                
            log_progress(f"  已保存批次数据到: {saved_path}，共{len(batch_df)}行", log_file)
            
            # 清空内存
            all_rows = []
    
    # 合并所有批次数据
    log_progress("合并所有批次数据...", log_file)
    batch_files = []
    
    # 检查parquet文件
    parquet_files = [f for f in os.listdir(output_dir) if f.startswith("attention_batch_") and f.endswith(".parquet")]
    if parquet_files:
        batch_files = parquet_files
        file_format = "parquet"
        log_progress(f"找到{len(parquet_files)}个parquet批次文件", log_file)
    else:
        # 检查csv文件
        csv_files = [f for f in os.listdir(output_dir) if f.startswith("attention_batch_") and f.endswith(".csv")]
        batch_files = csv_files
        file_format = "csv"
        log_progress(f"找到{len(csv_files)}个csv批次文件", log_file)
    
    if not batch_files:
        log_progress("警告: 未找到任何批次文件", log_file)
        return None
    
    # 按批次号排序
    def extract_batch_number(filename):
        """从文件名中提取批次号"""
        try:
            # 文件名格式: attention_batch_0.parquet 或 attention_batch_0.csv
            parts = filename.split("_")
            if len(parts) >= 3 and parts[0] == "attention" and parts[1] == "batch":
                batch_num_str = parts[2].split(".")[0]  # 去掉扩展名
                return int(batch_num_str)
            else:
                return 0  # 如果无法解析，返回0
        except (ValueError, IndexError):
            return 0  # 如果解析失败，返回0
    
    # 添加调试信息
    log_progress(f"批次文件列表: {batch_files}", log_file)
    
    batch_files.sort(key=extract_batch_number)
    log_progress(f"排序后的批次文件: {batch_files}", log_file)
    
    total_rows = 0
    combined_df = pd.DataFrame()
    
    # 显示合并进度
    merge_pbar = create_progress_bar(len(batch_files), desc="合并批次")
    
    for batch_file in batch_files:
        batch_path = os.path.join(output_dir, batch_file)
        try:
            batch_df = load_dataframe(batch_path)
            combined_df = pd.concat([combined_df, batch_df], ignore_index=True)
            total_rows += len(batch_df)
            # 立即释放内存
            del batch_df
            merge_pbar.update(1)
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
            log_progress(f"警告: 加载批次文件 {batch_file} 时出错: {e}", log_file)
    
    merge_pbar.close()
    
    # 保存最终结果
    log_progress(f"保存最终结果，总计{total_rows}行数据...", log_file)
    
    # 设置输出文件路径
    layer_num = layer_key.split('_')[-1]
    final_csv_path = os.path.join(output_dir, f"whole_slice_attention_layer_{layer_num}.csv")
    combined_df.to_csv(final_csv_path, index=False)
    log_progress(f"已合并所有数据并保存到CSV: {final_csv_path}", log_file)
    
    final_path = final_csv_path
    
    # 清理临时批次文件
    log_progress("清理临时批次文件...", log_file)
    for batch_file in batch_files:
        try:
            os.remove(os.path.join(output_dir, batch_file))
        except (FileNotFoundError, PermissionError, OSError) as e:
            log_progress(f"警告: 删除临时文件 {batch_file} 时出错: {e}", log_file)
    log_progress("已清理临时批次文件", log_file)
    
    # 计算处理速度
    elapsed = time.time() - start_time
    spots_per_second = total_centers / elapsed if elapsed > 0 else 0
    log_progress(f"完成全部{total_centers}个spots的处理，总耗时: {elapsed:.1f}秒，平均速度: {spots_per_second:.2f}spots/秒", log_file)
    
    return final_path

def export_optimized_data_structures(output_dir, attention_df, coords_df, log_file):
    """导出优化的数据结构，用于高效分析"""
    log_progress("创建优化的数据结构...", log_file)
    
    # 1. Spot-to-Spot注意力矩阵
    log_progress("  计算Spot-to-Spot注意力矩阵...", log_file)
    start_time = time.time()
    spot_to_spot = attention_df.groupby(['center_global_idx', 'neighbor_global_idx'])['attn_sum_norm'].sum().reset_index()
    spot_to_spot.columns = ['source_idx', 'target_idx', 'attention_weight']
    elapsed = time.time() - start_time
    
    # 保存
    spot_to_spot_path = os.path.join(output_dir, "spot_to_spot_attention")
    spot_to_spot_path = save_dataframe(spot_to_spot, spot_to_spot_path + ".csv")
        
    log_progress(f"  已保存Spot-to-Spot矩阵到: {spot_to_spot_path}，共{len(spot_to_spot)}行，用时: {elapsed:.1f}秒", log_file)
    
    # 2. 基因级别交互
    log_progress("  计算基因级别交互...", log_file)
    start_time = time.time()
    gene_interactions = attention_df[
        ['center_global_idx', 'neighbor_global_idx', 'q_gene_symbol', 'kv_gene_symbol', 'attn_score']
    ].dropna(subset=['q_gene_symbol', 'kv_gene_symbol'])
    
    # 只保留有意义的基因符号（非空字符串）
    gene_interactions = gene_interactions[
        (gene_interactions['q_gene_symbol'].str.len() > 0) & 
        (gene_interactions['kv_gene_symbol'].str.len() > 0)
    ]
    
    # 计算每对基因的平均注意力
    gene_pair_avg = gene_interactions.groupby(['q_gene_symbol', 'kv_gene_symbol'])['attn_score'].mean().reset_index()
    gene_pair_avg.columns = ['query_gene', 'key_gene', 'avg_attention']
    elapsed = time.time() - start_time
    
    # 保存
    gene_pair_path = os.path.join(output_dir, "gene_pair_attention")
    gene_pair_path = save_dataframe(gene_pair_avg, gene_pair_path + ".csv")
        
    log_progress(f"  已保存基因对平均注意力到: {gene_pair_path}，共{len(gene_pair_avg)}行，用时: {elapsed:.1f}秒", log_file)
    
    # 3. 每个spot的空间邻居（便于快速查找）
    log_progress("  计算空间邻居映射...", log_file)
    start_time = time.time()
    spot_neighbors = spot_to_spot.groupby('source_idx')['target_idx'].apply(list).reset_index()
    spot_neighbors.columns = ['spot_idx', 'neighbor_indices']
    elapsed = time.time() - start_time
    
    # 保存为pickle（保留列表结构）
    neighbors_path = os.path.join(output_dir, "spot_neighbors.pkl")
    
    try:
        spot_neighbors.to_pickle(neighbors_path)
        log_progress(f"  已保存空间邻居映射到: {neighbors_path}，共{len(spot_neighbors)}个spots，用时: {elapsed:.1f}秒", log_file)
    except (pickle.PickleError, IOError, OSError) as e:
        log_progress(f"  警告: 保存空间邻居映射为pickle失败: {e}", log_file)
        # 备用方案：保存为CSV（会丢失列表结构）
        fallback_path = os.path.join(output_dir, "spot_neighbors_fallback.csv")
        
        # 将列表转换为字符串
        spot_neighbors['neighbor_indices_str'] = spot_neighbors['neighbor_indices'].apply(lambda x: ','.join(map(str, x)))
        spot_neighbors[['spot_idx', 'neighbor_indices_str']].to_csv(fallback_path, index=False)
        
        log_progress(f"  已保存简化的空间邻居映射到: {fallback_path}", log_file)
        neighbors_path = fallback_path
    
    # 返回所有路径
    return {
        "spot_to_spot": spot_to_spot_path,
        "gene_pairs": gene_pair_path,
        "neighbors": neighbors_path
    }

def export_spatial_coordinates(adata, output_path, log_file):
    """导出空间坐标信息"""
    log_progress("导出空间坐标信息...", log_file)
    
    # 创建包含坐标和聚类信息的DataFrame
    coords_df = pd.DataFrame({
        'spot_idx': range(adata.n_obs),
        'x': adata.obsm['spatial'][:, 0],
        'y': adata.obsm['spatial'][:, 1],
    })
    
    # 添加聚类信息（如果有）
    if 'ground_truth' in adata.obs:
        coords_df['cluster'] = adata.obs['ground_truth'].values
        log_progress(f"  包含聚类信息: {coords_df['cluster'].nunique()}个聚类", log_file)
    
    # 保存为CSV
    coords_df.to_csv(output_path, index=False)
    log_progress(f"空间坐标信息已保存到: {output_path}，共{len(coords_df)}个spots", log_file)
    return output_path



def main(base_dir, h5ad_path, truth_path, vocab_path, attn_pkl_path, use_layer=5, batch_size=50):
    """主函数：处理全切片空间注意力数据"""
    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # output_dir = f"./HBRC1/whole_slice_data_{timestamp}"
    output_dir = f"./PDAC/whole_slice_data_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(output_dir, "processing_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"空间注意力数据处理日志 - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"参数设置:\n")
        f.write(f"  base_dir: {base_dir}\n")
        f.write(f"  h5ad_path: {h5ad_path}\n")
        f.write(f"  truth_path: {truth_path}\n")
        f.write(f"  vocab_path: {vocab_path}\n")
        f.write(f"  attn_pkl_path: {attn_pkl_path}\n")
        f.write(f"  use_layer: {use_layer}\n")
        f.write(f"  tqdm进度条: {'可用' if TQDM_AVAILABLE else '不可用'}\n")
        f.write("\n")
    
    # 记录开始时间
    total_start_time = time.time()
    log_progress("=================== 开始处理 ===================", log_file)
    log_progress(f"输出目录: {output_dir}", log_file)
    
    # 加载基础数据
    log_progress("加载anndata...", log_file)
    load_start = time.time()
    adata = sc.read_h5ad(h5ad_path)
    load_elapsed = time.time() - load_start
    log_progress(f"加载anndata完成，用时: {load_elapsed:.1f}秒", log_file)
    
    #### hbrc ####
    # try:
    #     df_meta = pd.read_csv(truth_path, header=0)
    #     df_meta_layer = df_meta["ground_truth"]
    #     adata.obs['ground_truth'] = df_meta_layer.values
    #     log_progress("已加载ground truth", log_file)
    # except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
    #     log_progress(f"警告: 无法加载ground truth: {e}", log_file)

    #### pdac ####
    df_meta_layer = pd.read_csv(truth_path)["Region"]
    adata.obs['ground_truth'] = df_meta_layer.values

    
    adata.obs['spot_idx'] = np.arange(adata.n_obs)
    log_progress(f"Total spots (adata): {adata.n_obs}", log_file)
    
    adata_out_path = os.path.join(output_dir, "adata_with_metadata.h5ad")
    adata.write_h5ad(adata_out_path)
    log_progress(f"Annotated adata已保存到: {adata_out_path}", log_file)
    
    # 导出空间坐标
    coords_path = os.path.join(output_dir, "spatial_coordinates.csv")
    export_spatial_coordinates(adata, coords_path, log_file)
    
    # 加载vocab
    log_progress("构建基因ID到符号的映射...", log_file)
    vocab_start = time.time()
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    id2sym = build_id2sym(vocab)
    vocab_elapsed = time.time() - vocab_start
    log_progress(f"vocab构建完成，大小: {len(id2sym)}，用时: {vocab_elapsed:.1f}秒", log_file)
    
    # 加载attention packs
    log_progress("读取attention数据...", log_file)
    attn_start = time.time()
    with open(attn_pkl_path, "rb") as f:
        packs = pickle.load(f)
    attn_elapsed = time.time() - attn_start
    log_progress(f"attention数据读取完成，用时: {attn_elapsed:.1f}秒", log_file)
    
    if not isinstance(packs, list) or len(packs) == 0:
        raise ValueError("attention packs 格式异常，期望为非空 list")
    
    # 创建全局索引到名称的映射
    global_idx_to_name = {i: str(name) for i, name in enumerate(adata.obs_names)}
    
    # 处理全切片空间注意力数据
    layer_key = f"context_encoder_layer_{use_layer}"
    attention_path = process_all_spots_efficiently(
        packs=packs,
        layer_key=layer_key,
        adata=adata,
        id2sym=id2sym,
        global_idx_to_name=global_idx_to_name,
        output_dir=output_dir,
        log_file=log_file,
        batch_size=batch_size
    )
    
    # 加载处理后的数据并创建优化的数据结构
    log_progress("加载处理后的数据并创建优化的数据结构...", log_file)
    opt_start = time.time()
    attention_df = load_dataframe(attention_path)
    optimized_paths = export_optimized_data_structures(output_dir, attention_df, adata.obs, log_file)
    opt_elapsed = time.time() - opt_start
    log_progress(f"数据结构优化完成，用时: {opt_elapsed:.1f}秒", log_file)
    
    # 导出配体-受体示例数据库
    lr_db_path = os.path.join(output_dir, "lr_database.csv")
    # export_lr_database(lr_db_path, log_file)
    
    # 保存配置信息
    config = {
        "h5ad_path": h5ad_path,
        "truth_path": truth_path,
        "vocab_path": vocab_path,
        "attn_pkl_path": attn_pkl_path,
        "use_layer": use_layer,
        "output_dir": output_dir,
        "adata_path": adata_out_path,
        "coords_path": coords_path,
        "attention_path": attention_path,
        "optimized_paths": optimized_paths,
        "lr_db_path": lr_db_path,
        "timestamp": timestamp,
        "processing_log": log_file,
    }
    
    config_path = os.path.join(output_dir, "export_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 计算总用时
    total_elapsed = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = ""
    if hours > 0:
        time_str += f"{int(hours)}小时 "
    if minutes > 0 or hours > 0:
        time_str += f"{int(minutes)}分钟 "
    time_str += f"{seconds:.1f}秒"
    
    log_progress(f"=================== 处理完成 ===================", log_file)
    log_progress(f"全切片数据处理完成！总用时: {time_str}", log_file)
    log_progress(f"所有数据已保存到: {output_dir}", log_file)
    log_progress(f"配置信息: {config_path}", log_file)
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增强版空间注意力数据导出工具")
    parser.add_argument('--base_dir', type=str, help='基础目录路径')
    parser.add_argument('--h5ad_path', type=str, help='H5AD文件路径')
    parser.add_argument('--truth_path', type=str, help='真实标签文件路径')
    parser.add_argument('--vocab_path', type=str, help='vocab.json文件路径')
    parser.add_argument('--attn_pkl_path', type=str, help='attention pkl文件路径')
    parser.add_argument('--use_layer', type=int, default=5, help='使用的层号 (默认: 5)')
    parser.add_argument('--batch_size', type=int, default=50, help='批处理大小 (默认: 50)')
    
    args = parser.parse_args()
    
    # 如果命令行参数未提供，则使用默认值
    if not all([args.base_dir, args.h5ad_path, args.truth_path, args.vocab_path, args.attn_pkl_path]):
        print("使用默认路径...")
        # 基本路径设置
        # base_dir = "/home/junning/projectnvme/ST/project-20-contrast-organ/Inference_embeddings/HBRC_clsrecon_nosemantic_finetune1"
        # h5ad_path = '/home/junning/projectnvme/ST/h5ad/HBRC/human-breast-cancer.h5ad'
        # truth_path = '/home/junning/projectnvme/ST/Data/HBRC/hbrc_truth.csv'


        base_dir = "/home/junning/projectnvme/ST/project-20-contrast-organ/Inference_embeddings/PDAC/PDAC_clsrecon_nosemantic_fintuneembedding_cls2genes_submodel"
        h5ad_path = '/home/junning/projectnvme/ST/Data/PDAC/pdac.h5ad'
        truth_path =  '/home/junning/projectnvme/ST/Data/PDAC/PDAC_truth.csv'


        vocab_path = "/home/junning/projectnvme/ST/Data/Get_Embedding/vocab.json"
        attn_pkl_path = os.path.join(base_dir, "context_attention_scores.pkl")
        use_layer = 5
        batch_size = 50
    else:
        base_dir = args.base_dir
        h5ad_path = args.h5ad_path
        truth_path = args.truth_path
        vocab_path = args.vocab_path
        attn_pkl_path = args.attn_pkl_path
        use_layer = args.use_layer
        batch_size = args.batch_size
    
    # 运行主函数
    output_dir = main(
        base_dir=base_dir,
        h5ad_path=h5ad_path,
        truth_path=truth_path,
        vocab_path=vocab_path,
        attn_pkl_path=attn_pkl_path,
        use_layer=use_layer,
        batch_size=batch_size
    )