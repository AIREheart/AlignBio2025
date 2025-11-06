"""
Data processing utilities for PETase Predictor.

Functions:
- load_workbook()
- detect_and_merge_embeddings()
- build_datasets() -> PyTorch Dataset objects
- metrics utilities (ndcg)
"""

import os, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from typing import Optional, Tuple, List

AA = list("ARNDCQEGHILKMFPSTWYV")
AA_IDX = {a: i+1 for i,a in enumerate(AA)}  # 0 reserved for padding

def load_workbook(path: str, main_sheet: Optional[str]=None) -> dict:
    xls = pd.read_excel(path, sheet_name=None)
    if main_sheet is None:
        # choose the largest sheet by rows
        main_sheet = max(xls.keys(), key=lambda k: len(xls[k]))
    return {"xls": xls, "main_sheet": main_sheet}

def detect_seq_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        sample = df[c].dropna().astype(str).iloc[:20].tolist() if df[c].dropna().shape[0]>0 else []
        if sample and all(len(s) > 5 and set(s.upper()).issubset(set("ACDEFGHIKLMNPQRSTVWYBXZJUO-")) for s in sample):
            return c
    return None

def safe_numeric_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            try:
                pd.to_numeric(df[c], errors='coerce')
                return c
            except Exception:
                continue
    # fallback: find column with 'tm' in name
    for c in df.columns:
        if 'tm' in str(c).lower():
            return c
    return None

def featurize_seq_onehot(seq: str, L_max: int):
    seq = (seq or "").upper()
    arr = np.zeros((L_max, len(AA)+1), dtype=np.float32)  # +1 for padding idx 0
    for i,ch in enumerate(seq[:L_max]):
        arr[i, AA_IDX.get(ch, 0)] = 1.0
    return arr

def build_local_adj(L_max: int, window: int = 6):
    A = np.zeros((L_max, L_max), dtype=np.float32)
    for i in range(L_max):
        for j in range(max(0,i-window), min(L_max, i+window+1)):
            A[i,j] = 1.0
    for i in range(L_max-1):
        A[i,i+1] = 1.0; A[i+1,i]=1.0
    return A

def create_grouped_splits(df: pd.DataFrame, group_col: Optional[str]=None, train_frac: float=0.7, seed: int=42):
    if group_col is None or group_col not in df.columns:
        groups = df.index.astype(str)
    else:
        groups = df[group_col].astype(str).fillna("unknown")
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
   train_idx, rest_idx = next(splitter.split(df, groups=groups))
    rest = df.iloc[rest_idx]
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    vrel, trel = next(splitter2.split(rest, groups=rest[groups.name])) if isinstance(groups, pd.Series) else next(splitter2.split(rest, groups=rest.index))
    val_idx = rest_idx[vrel]; test_idx = rest_idx[trel]
    out = df.copy()
    out["_split"] = "train"
    out.loc[out.index.isin(val_idx), "_split"] = "val"
    out.loc[out.index.isin(test_idx), "_split"] = "test"
    return out

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: Optional[int]=None) -> float:
    if len(y_true) == 0:
        return 0.0
    if k is None:
        k = len(y_true)
    order = np.argsort(-y_score)
    y_sorted = np.take(y_true, order[:k])
    gains = 2**(y_sorted) - 1
    discounts = np.log2(np.arange(2, k+2))
    dcg = (gains / discounts).sum()
    ideal = np.sort(y_true)[::-1][:k]
    idcg = ((2**ideal - 1) / np.log2(np.arange(2, k+2))).sum()
    return float(dcg / (idcg + 1e-8))

# Utility to load per-variant embeddings saved as numpy array + mapping (id list)
def load_embeddings_np(emb_npy_path: str, emb_map_csv: Optional[str]=None):
    emb = np.load(emb_npy_path, allow_pickle=True)
    # emb expected shape: (N, dim) or dict {'ids': [...], 'vecs': ...}
    if isinstance(emb, np.ndarray):
        return emb
    if isinstance(emb, dict) or hasattr(emb, 'tolist'):
        return emb
    return emb
