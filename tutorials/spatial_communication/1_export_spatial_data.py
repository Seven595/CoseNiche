""" attention data export - Optimizationprocessing - export Optimization for analysis and can - - Parquet of - of gene filtering:HLAgene,filter description (2025-10-29): - gene filtering, all of gene - gene (HLA-*, MIR*) - filtergene, gene, RNA - 60.6% of gene (vs59.9%), all 38 HLAgene """

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
    print("Warning: not foundtqdm,")


# of Warning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# gene filtering (by gene filteringplot)
EXTRA_EXCLUDE_PREFIXES = ("AC", "AL", "SNOR", "SCARNA", "LINC")
DROP_WITH_DOT_OR_DASH = False  # :filter all
EXTRA_EXCLUDE_REGEX = None

# ------------------ ------------------
def get_time_estimate(start_time, current_step, total_steps):
    """compute"""
    elapsed = time.time() - start_time
    if current_step == 0:
        return "compute in ..."
    
    steps_per_second = current_step / elapsed
    remaining_seconds = (total_steps - current_step) / steps_per_second
    
    # for can format
    if remaining_seconds < 60:
        return f"{remaining_seconds:.1f}sec"
    elif remaining_seconds < 3600:
        return f"{remaining_seconds/60:.1f}min"
    else:
        return f"{remaining_seconds/3600:.1f}h"

def create_progress_bar(total, desc="processing in "):
    """createProgress bar"""
    if TQDM_AVAILABLE:
        return tqdm(total=total, desc=desc, unit="spot", ncols=100, 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    else:
        # of
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
                    print(f"{self.desc}: {self.n}/{self.total} [{elapsed:.1f}s ]")
                    
            def set_postfix(self, **kwargs):
                # postfix
                pass
                
            def close(self):
                elapsed = time.time() - self.start_time
                print(f"{self.desc}: Completed {self.n}/{self.total} [{elapsed:.1f}s total]")
                
        return SimpleProgressBar(total, desc)

def log_progress(message, log_file=None):
    """"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # if Loggingfile,Logging
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')

# ------------------ save and load ------------------
def save_dataframe(df, path, format_hint="auto"):
    """Save Data Frame,processing different format"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # fileformat
    if format_hint == "auto":
        if path.endswith(".parquet"):
            format_hint = "parquet"
        elif path.endswith(".csv"):
            format_hint = "csv"
        else:
            # default CSV
            format_hint = "csv"
            if not path.endswith(".csv"):
                path = path + ".csv"
    
    # format Save
    df.to_csv(path, index=False)
    return path

def load_dataframe(path):
    """load Data Frame,processing different format"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"file in : {path}")
    
    return pd.read_csv(path)

# ------------------ ------------------
def to_np(x):
    """ for Num Py"""
    if isinstance(x, np.ndarray): return x
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.asarray(x)

def clean_symbol(s: str) -> str:
    """gene symbols"""
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

DOT_DASH_PATTERN = re.compile(r"[.\-\u00B7\u2219\u2010\u2011\u2012\u2013\u2014\u2015]")

def build_id2sym(vd: Dict[Any, Any]) -> Dict[int, str]:
    """buildID to of """
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

def ids_to_clean_symbols(gids: Sequence[Any], id2sym: Dict[int, str]) -> List[str]:
    """IDcolumntable for of columntable"""
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
    """ gene symbolsgene () by gene filteringplot: Step1: check - HLA- and MIR of gene Step2: check - of gene Step3: check - of gene Step4: Prefix blacklistCheck - before of gene Step5: check - of """
    symbol = clean_symbol(symbol)
    if not symbol:
        return False
    
    symbol_lower = symbol.lower()
    
    # Step1: check - gene before (HLA-, MIR)
    if symbol.startswith(('HLA-', 'MIR')):
        return True
    
    # Step2: check - of gene
    if "." in symbol:
        return False
    
    # Step3: check - filtergene, RNA
    regex_patterns = [
        r'^[A-Z]{2}\d+\.\d+$',      # AC012345.1 (gene)
        r'^ABBA\d+\.\d+$',           # ABBA01000934.1
        r'-AS\d*$',                  # A1BG-AS1 (RNA)
        r'^RP\d+-\w+\.\d+$',         # RP11-206L10.2 (gene)
        r'^CTD-\w+\.\d+$',           # CTDgene
        r'^CTC-\w+\.\d+$'            # CTCgene
    ]
    for pattern in regex_patterns:
        if re.match(pattern, symbol):
            return False
    
    # Step4: Prefix blacklistCheck (AC, AL, SNOR, SCARNA, LINC)
    prefixes = tuple(p.lower() for p in (extra_exclude_prefixes or []))
    if prefixes and any(symbol_lower.startswith(px) for px in prefixes):
        return False
    
    # Step5: check (if)
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
    """filtergeneIDcolumntable,gene"""
    valid_indices = []
    filtered_ids = []
    filtered_symbols = []
    
    # columntable
    original_symbols = []
    for gid in gene_ids:
        try:
            sym = id2sym.get(int(gid), str(gid))
            original_symbols.append(sym)
        except Exception:
            original_symbols.append(str(gid))
    
    # Output
    if verbose:
        log_progress(f"Startfilter,number of genes: {len(gene_ids)}")
        log_progress(f" before 5genesID:")
        for i in range(min(5, len(gene_ids))):
            gid = gene_ids[i]
            sym = original_symbols[i]
            is_filtered = not is_coding_gene(sym, drop_with_dot_or_dash, extra_exclude_prefixes, extra_exclude_regex)
            log_progress(f" [{i}] ID={gid}, Symbol={sym}, filter: {is_filtered}")
        
        # in of number of genes
        non_coding_count = sum(1 for sym in original_symbols if not is_coding_gene(sym, drop_with_dot_or_dash, extra_exclude_prefixes, extra_exclude_regex))
        log_progress(f" in gene: {non_coding_count}/{len(original_symbols)} ({non_coding_count/len(original_symbols)*100:.1f}%)")
    
    # rowsfilter
    for i, gid in enumerate(gene_ids):
        try:
            sym = original_symbols[i]
            # if gene,
            if is_coding_gene(sym, drop_with_dot_or_dash, extra_exclude_prefixes, extra_exclude_regex):
                valid_indices.append(i)
                filtered_ids.append(gid)
                filtered_symbols.append(clean_symbol(sym))
        except Exception as e:
            if verbose:
                log_progress(f"Warning: processinggeneID {gid} : {e}")
    
    # Outputfilter
    if verbose:
        log_progress(f"filter Completed,number of genes: {len(filtered_ids)}/{len(gene_ids)} ({len(filtered_ids)/len(gene_ids)*100:.1f}%)")
        if filtered_symbols:
            log_progress(f" after filtering  before 5 gene symbols: {filtered_symbols[:5]}")
    
    return filtered_ids, valid_indices, filtered_symbols

# ------------------ attention dataprocessing ------------------
def flatten_packs_to_centers(
    packs: List[Dict[str, Any]],
    layer_key: str,
    global_idx_to_name: Optional[Dict[int,str]] = None,
    log_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """ from packs[layer_key]['spatial_cross_attention'] for per-center. weights for [C, Q, K] or [C, H, Q, K] (). """
    log_progress(f"attention packs (: {layer_key})...", log_file)
    flat = []
    total_packs = len(packs)
    
    for pi, pack in enumerate(packs):
        if (pi + 1) % 10 == 0 or pi == 0 or pi == total_packs - 1:
            log_progress(f"processingpack {pi+1}/{total_packs}...", log_file)
        
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

        # Note.
        if W.ndim == 4:
            # [C, H, Q, K] -> mean over H
            W = W.mean(axis=1)
        elif W.ndim != 3:
            raise ValueError(f"Unexpected weights shape: {W.shape}")

        C, q_max_in_W, Kmax = W.shape

        # align slices
        if isinstance(slices_per_center, list):
            if len(slices_per_center) == C and (len(slices_per_center) == 0 or isinstance(slices_per_center[0], list)):
                slices_aligned = slices_per_center
            elif len(slices_per_center) > 0 and isinstance(slices_per_center[0], dict):
                # columntable, to each in
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
    
    log_progress(f" for {len(flat)} center spots", log_file)
    return flat

def neighbor_groups_for_center(slices_i: List[Dict[str, int]]) -> List[Tuple[int, List[Tuple[int,int]]]]:
    """ in of """
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
    """ in of Top-K"""
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
    
    # check:topkactualavailable of count
    actual_topk = min(topk, len(cols))
    
    order = np.argsort(-scores)[:actual_topk]
    return cols[order], scores[order]

def process_all_spots_efficiently(packs, layer_key, adata, id2sym, global_idx_to_name, output_dir, log_file, batch_size=50):
    """processing and of all spots"""
    log_progress(f"Start processing of all spots (:{layer_key})...", log_file)
    
    # all center spots of attention data
    flat_centers = flatten_packs_to_centers(
        packs=packs,
        layer_key=layer_key,
        global_idx_to_name=global_idx_to_name,
        log_file=log_file
    )
    
    total_centers = len(flat_centers)
    log_progress(f"Found{total_centers} center spots", log_file)
    
    # create for Saveall of Data Frame
    all_rows = []
    # batch_size for parameters
    
    # batch
    total_batches = (total_centers - 1) // batch_size + 1
    start_time = time.time()
    
    for batch_idx in range(0, total_centers, batch_size):
        batch_num = batch_idx // batch_size + 1
        log_progress(f"processingbatch {batch_num}/{total_batches}...", log_file)
        
        # Note.
        if batch_idx > 0:
            elapsed = time.time() - start_time
            spots_per_second = batch_idx / elapsed
            remaining_spots = total_centers - batch_idx
            remaining_seconds = remaining_spots / spots_per_second
            
            if remaining_seconds < 60:
                time_str = f"{remaining_seconds:.1f}sec"
            elif remaining_seconds < 3600:
                time_str = f"{remaining_seconds/60:.1f}min"
            else:
                time_str = f"{remaining_seconds/3600:.1f}h"
                
            log_progress(f" : {time_str}", log_file)
        
        end_idx = min(batch_idx + batch_size, total_centers)
        batch_centers = flat_centers[batch_idx:end_idx]
        
        # for before batchcreateProgress bar
        pbar = create_progress_bar(len(batch_centers), desc=f"Batch {batch_num}/{total_batches}")
        
        # processing each batch in  of spot
        for i, center_entry in enumerate(batch_centers):
            current_idx = batch_idx + i
            
            # Progress bar
            pbar.update(1)
            pbar.set_postfix(spot=current_idx+1, total=total_centers)
            
            # spot of
            W_i = center_entry["weights"]              # [q_max,Kmax]
            qlen = int(center_entry["q_valid_len"])
            slices_i = center_entry["kv_source_slices"]
            btlg = center_entry.get("batch_local_to_global", None)
            kv_gene_ids = center_entry.get("kv_gene_ids", None)
            query_gene_ids_arr = center_entry.get("query_gene_ids", None)
            center_global = int(center_entry["center_global_idx"])
            center_name = center_entry.get("center_name", "")
            Kmax = W_i.shape[1]

            # processingquerygene
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
                
                # filterquerygene
                for q, gid in enumerate(query_genes):
                    if gid is None:
                        continue
                        
                    if id2sym is not None:
                        sym = id2sym.get(gid, "")
                        # Checkgene
                        if is_coding_gene(sym):
                            filtered_q_indices.append(q)
                            filtered_q_genes.append(gid)
                    else:
                        filtered_q_indices.append(q)
                        filtered_q_genes.append(gid)
            else:
                # querygene,all
                filtered_q_indices = list(range(qlen))
                filtered_q_genes = [None] * qlen
            
            # Note.
            groups = neighbor_groups_for_center(slices_i)
            
            # processing each querygene and
            for q_idx, q in enumerate(filtered_q_indices):
                # querygene
                q_gid = filtered_q_genes[q_idx] if filtered_q_genes[q_idx] is not None else ""
                q_sym = ""
                if q_gid and id2sym is not None:
                    q_sym = id2sym.get(q_gid, "")

                # computeneighbors of total and
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
                    
                    if ssum > 0:  # of
                        neighbor_items.append(dict(
                            neighbor_local_row=int(nl),
                            neighbor_global_idx=nei_global,
                            neighbor_name=nei_name,
                            attn_sum=float(ssum)
                        ))
                        total_sum += ssum

                # Note.
                for it in neighbor_items:
                    if total_sum > 0:
                        it["attn_sum_norm"] = it["attn_sum"] / total_sum
                    else:
                        it["attn_sum_norm"] = 0.0

                # top
                if neighbor_items:
                    neighbor_items.sort(key=lambda x: x["attn_sum"], reverse=True)
                    top_neighbors = neighbor_items[:7]  # before 7neighbors
                    
                    # neighborstop gene
                    for it in top_neighbors:
                        nl = it["neighbor_local_row"]
                        segs = next((s for n, s in groups if n == nl), [])
                        
                        # of top
                        cols, scores = top_k_keys_in_neighbor_for_query(W_i, segs, q_index=q, topk=10)
                        
                        if cols.size > 0:
                            # processing of gene
                            for c_abs, sc in zip(cols, scores):
                                kv_gid = ""
                                kv_sym = ""
                                
                                if kv_gene_ids is not None:
                                    gid = int(kv_gene_ids[int(c_abs)])
                                    if gid >= 0:
                                        kv_gid = gid
                                        if id2sym is not None:
                                            kv_sym = id2sym.get(kv_gid, "")
                                
                                # rows of
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
                            # rowsneighbor level of
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
        
        # Progress bar
        pbar.close()
        
        # processingsave in
        if all_rows:
            # save
            batch_df = pd.DataFrame(all_rows)
            batch_path = os.path.join(output_dir, f"attention_batch_{batch_idx//batch_size}")
            
            # save,processing format
            saved_path = save_dataframe(batch_df, batch_path + ".csv")
                
            log_progress(f" saved to : {saved_path},{len(batch_df)}rows", log_file)
            
            # Note.
            all_rows = []
    
    # all
    log_progress(" all...", log_file)
    batch_files = []
    
    # Checkparquetfile
    parquet_files = [f for f in os.listdir(output_dir) if f.startswith("attention_batch_") and f.endswith(".parquet")]
    if parquet_files:
        batch_files = parquet_files
        file_format = "parquet"
        log_progress(f" found {len(parquet_files)} parquetfile", log_file)
    else:
        # Checkcsvfile
        csv_files = [f for f in os.listdir(output_dir) if f.startswith("attention_batch_") and f.endswith(".csv")]
        batch_files = csv_files
        file_format = "csv"
        log_progress(f" found {len(csv_files)} csvfile", log_file)
    
    if not batch_files:
        log_progress("Warning: not foundfile", log_file)
        return None
    
    # by
    def extract_batch_number(filename):
        """ from file in """
        try:
            # fileformat: attention_batch_0.parquet or attention_batch_0.csv
            parts = filename.split("_")
            if len(parts) >= 3 and parts[0] == "attention" and parts[1] == "batch":
                batch_num_str = parts[2].split(".")[0]  # Note.
                return int(batch_num_str)
            else:
                return 0  # if,0
        except (ValueError, IndexError):
            return 0  # if failed,0
    
    # Note.
    log_progress(f"filecolumntable: {batch_files}", log_file)
    
    batch_files.sort(key=extract_batch_number)
    log_progress(f" after of file: {batch_files}", log_file)
    
    total_rows = 0
    combined_df = pd.DataFrame()
    
    # Note.
    merge_pbar = create_progress_bar(len(batch_files), desc="cell_type")
    
    for batch_file in batch_files:
        batch_path = os.path.join(output_dir, batch_file)
        try:
            batch_df = load_dataframe(batch_path)
            combined_df = pd.concat([combined_df, batch_df], ignore_index=True)
            total_rows += len(batch_df)
            # Note.
            del batch_df
            merge_pbar.update(1)
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
            log_progress(f"Warning: loadfile {batch_file} : {e}", log_file)
    
    merge_pbar.close()
    
    # save
    log_progress(f"save,Total{total_rows}rows...", log_file)
    
    # Outputfile path
    layer_num = layer_key.split('_')[-1]
    final_csv_path = os.path.join(output_dir, f"whole_slice_attention_layer_{layer_num}.csv")
    combined_df.to_csv(final_csv_path, index=False)
    log_progress(f" all save to CSV: {final_csv_path}", log_file)
    
    final_path = final_csv_path
    
    # file
    log_progress("file...", log_file)
    for batch_file in batch_files:
        try:
            os.remove(os.path.join(output_dir, batch_file))
        except (FileNotFoundError, PermissionError, OSError) as e:
            log_progress(f"Warning: file {batch_file} : {e}", log_file)
    log_progress("file", log_file)
    
    # computeprocessing
    elapsed = time.time() - start_time
    spots_per_second = total_centers / elapsed if elapsed > 0 else 0
    log_progress(f"Completedall{total_centers} spots of processing,total: {elapsed:.1f}sec,: {spots_per_second:.2f}spots/sec", log_file)
    
    return final_path

def export_optimized_data_structures(output_dir, attention_df, coords_df, log_file):
    """exportOptimization of, for analysis"""
    log_progress("createOptimization of...", log_file)
    
    # 1. Spot-to-Spot
    log_progress(" Computing spot-to-spot attention...", log_file)
    start_time = time.time()
    spot_to_spot = attention_df.groupby(['center_global_idx', 'neighbor_global_idx'])['attn_sum_norm'].sum().reset_index()
    spot_to_spot.columns = ['source_idx', 'target_idx', 'attention_weight']
    elapsed = time.time() - start_time
    
    # Save
    spot_to_spot_path = os.path.join(output_dir, "spot_to_spot_attention")
    spot_to_spot_path = save_dataframe(spot_to_spot, spot_to_spot_path + ".csv")
        
    log_progress(f" Saved Spot-to-Spot to : {spot_to_spot_path},{len(spot_to_spot)}rows,: {elapsed:.1f}sec", log_file)
    
    # 2. gene-level
    log_progress(" computegene-level...", log_file)
    start_time = time.time()
    gene_interactions = attention_df[
        ['center_global_idx', 'neighbor_global_idx', 'q_gene_symbol', 'kv_gene_symbol', 'attn_score']
    ].dropna(subset=['q_gene_symbol', 'kv_gene_symbol'])
    
    # of gene symbols (non-empty)
    gene_interactions = gene_interactions[
        (gene_interactions['q_gene_symbol'].str.len() > 0) & 
        (gene_interactions['kv_gene_symbol'].str.len() > 0)
    ]
    
    # computegene of
    gene_pair_avg = gene_interactions.groupby(['q_gene_symbol', 'kv_gene_symbol'])['attn_score'].mean().reset_index()
    gene_pair_avg.columns = ['query_gene', 'key_gene', 'avg_attention']
    elapsed = time.time() - start_time
    
    # Save
    gene_pair_path = os.path.join(output_dir, "gene_pair_attention")
    gene_pair_path = save_dataframe(gene_pair_avg, gene_pair_path + ".csv")
        
    log_progress(f" Savedgene pairs to : {gene_pair_path},{len(gene_pair_avg)}rows,: {elapsed:.1f}sec", log_file)
    
    # 3. each spot of ()
    log_progress(" compute...", log_file)
    start_time = time.time()
    spot_neighbors = spot_to_spot.groupby('source_idx')['target_idx'].apply(list).reset_index()
    spot_neighbors.columns = ['spot_idx', 'neighbor_indices']
    elapsed = time.time() - start_time
    
    # Save for pickle (columntable)
    neighbors_path = os.path.join(output_dir, "spot_neighbors.pkl")
    
    try:
        spot_neighbors.to_pickle(neighbors_path)
        log_progress(f" saved to : {neighbors_path},{len(spot_neighbors)} spots,: {elapsed:.1f}sec", log_file)
    except (pickle.PickleError, IOError, OSError) as e:
        log_progress(f" Warning: save for picklefailed: {e}", log_file)
        # :Save for CSV (columntable)
        fallback_path = os.path.join(output_dir, "spot_neighbors_fallback.csv")
        
        # columntable for
        spot_neighbors['neighbor_indices_str'] = spot_neighbors['neighbor_indices'].apply(lambda x: ','.join(map(str, x)))
        spot_neighbors[['spot_idx', 'neighbor_indices_str']].to_csv(fallback_path, index=False)
        
        log_progress(f" saved of to : {fallback_path}", log_file)
        neighbors_path = fallback_path
    
    # all path
    return {
        "spot_to_spot": spot_to_spot_path,
        "gene_pairs": gene_pair_path,
        "neighbors": neighbors_path
    }

def export_spatial_coordinates(adata, output_path, log_file):
    """exportspatial coordinates"""
    log_progress("exportspatial coordinates...", log_file)
    
    # createcontains and cluster information of Data Frame
    coords_df = pd.DataFrame({
        'spot_idx': range(adata.n_obs),
        'x': adata.obsm['spatial'][:, 0],
        'y': adata.obsm['spatial'][:, 1],
    })
    
    # cluster information (if)
    if 'ground_truth' in adata.obs:
        coords_df['cluster'] = adata.obs['ground_truth'].values
        log_progress(f"  containscluster information: {coords_df['cluster'].nunique()} cluster", log_file)
    
    # Save for CSV
    coords_df.to_csv(output_path, index=False)
    log_progress(f"spatial coordinatessaved to : {output_path},{len(coords_df)} spots", log_file)
    return output_path



def main(base_dir, h5ad_path, truth_path, vocab_path, attn_pkl_path, use_layer=5, batch_size=50):
    """:processingattention data"""
    # createOutput directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # output_dir = f"./HBRC1/whole_slice_data_{timestamp}"
    output_dir = f"./PDAC/whole_slice_data_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # createLoggingfile
    log_file = os.path.join(output_dir, "processing_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"attention dataprocessing Logging - Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"parameters:\n")
        f.write(f"  base_dir: {base_dir}\n")
        f.write(f"  h5ad_path: {h5ad_path}\n")
        f.write(f"  truth_path: {truth_path}\n")
        f.write(f"  vocab_path: {vocab_path}\n")
        f.write(f"  attn_pkl_path: {attn_pkl_path}\n")
        f.write(f"  use_layer: {use_layer}\n")
        f.write(f"  tqdmProgress bar: {'available' if TQDM_AVAILABLE else 'unavailable'}\n")
        f.write("\n")
    
    # Start
    total_start_time = time.time()
    log_progress("=================== Start processing ===================", log_file)
    log_progress(f"Output directory: {output_dir}", log_file)
    
    # load
    log_progress("loadanndata...", log_file)
    load_start = time.time()
    adata = sc.read_h5ad(h5ad_path)
    load_elapsed = time.time() - load_start
    log_progress(f"loadanndataCompleted,: {load_elapsed:.1f}sec", log_file)
    
    #### hbrc ####
    # try:
    #     df_meta = pd.read_csv(truth_path, header=0)
    #     df_meta_layer = df_meta["ground_truth"]
    #     adata.obs['ground_truth'] = df_meta_layer.values
    # log_progress("loadground truth", log_file)
    # except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
    # log_progress(f"Warning: loadground truth: {e}", log_file)

    #### pdac ####
    df_meta_layer = pd.read_csv(truth_path)["Region"]
    adata.obs['ground_truth'] = df_meta_layer.values

    
    adata.obs['spot_idx'] = np.arange(adata.n_obs)
    log_progress(f"Total spots (adata): {adata.n_obs}", log_file)
    
    adata_out_path = os.path.join(output_dir, "adata_with_metadata.h5ad")
    adata.write_h5ad(adata_out_path)
    log_progress(f"Annotated adataSaved to : {adata_out_path}", log_file)
    
    # exportspatial coordinates
    coords_path = os.path.join(output_dir, "spatial_coordinates.csv")
    export_spatial_coordinates(adata, coords_path, log_file)
    
    # loadvocab
    log_progress("buildgeneID to of...", log_file)
    vocab_start = time.time()
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    id2sym = build_id2sym(vocab)
    vocab_elapsed = time.time() - vocab_start
    log_progress(f"vocabbuildCompleted,size: {len(id2sym)},: {vocab_elapsed:.1f}sec", log_file)
    
    # loadattention packs
    log_progress("Readingattention...", log_file)
    attn_start = time.time()
    with open(attn_pkl_path, "rb") as f:
        packs = pickle.load(f)
    attn_elapsed = time.time() - attn_start
    log_progress(f"attentionreadingCompleted,: {attn_elapsed:.1f}sec", log_file)
    
    if not isinstance(packs, list) or len(packs) == 0:
        raise ValueError("attention packs formatinvalid,expected to benon-empty list")
    
    # create to of
    global_idx_to_name = {i: str(name) for i, name in enumerate(adata.obs_names)}
    
    # processingattention data
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
    
    # loadprocessing after of create Optimization of
    log_progress("loadprocessing after of create Optimization of...", log_file)
    opt_start = time.time()
    attention_df = load_dataframe(attention_path)
    optimized_paths = export_optimized_data_structures(output_dir, attention_df, adata.obs, log_file)
    opt_elapsed = time.time() - opt_start
    log_progress(f"OptimizationCompleted,: {opt_elapsed:.1f}sec", log_file)
    
    # exportligand-receptordatabase
    lr_db_path = os.path.join(output_dir, "lr_database.csv")
    # export_lr_database(lr_db_path, log_file)
    
    # save
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
    
    # computetotal
    total_elapsed = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = ""
    if hours > 0:
        time_str += f"{int(hours)}h "
    if minutes > 0 or hours > 0:
        time_str += f"{int(minutes)}min "
    time_str += f"{seconds:.1f}sec"
    
    log_progress(f"=================== processingCompleted ===================", log_file)
    log_progress(f"processing Completed！total: {time_str}", log_file)
    log_progress(f" all saved to : {output_dir}", log_file)
    log_progress(f": {config_path}", log_file)
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced spatial attention data export tool")
    parser.add_argument('--base_dir', type=str, help='base directorypath')
    parser.add_argument('--h5ad_path', type=str, help='H5ADfile path')
    parser.add_argument('--truth_path', type=str, help='truth labelsfile path')
    parser.add_argument('--vocab_path', type=str, help='vocab.jsonfile path')
    parser.add_argument('--attn_pkl_path', type=str, help='attention pklfile path')
    parser.add_argument('--use_layer', type=int, default=5, help='layer number to use (default: 5)')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size (default: 50)')
    
    args = parser.parse_args()
    
    # if rows Parameters not,default
    if not all([args.base_dir, args.h5ad_path, args.truth_path, args.vocab_path, args.attn_pkl_path]):
        print("Using default paths...")
        # path
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
    
    # rows
    output_dir = main(
        base_dir=base_dir,
        h5ad_path=h5ad_path,
        truth_path=truth_path,
        vocab_path=vocab_path,
        attn_pkl_path=attn_pkl_path,
        use_layer=use_layer,
        batch_size=batch_size
    )