"""
Data loading and preprocessing utilities for PETase Predictor.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import torch

AA = list("ARNDCQEGHILKMFPSTWYV")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}

def load_data(path):
    xls = pd.read_excel(path, sheet_name=None)
    main = xls.get("150petase")
    emb = xls.get("petase_embeddings")
    if emb is not None:
        # Merge on Name (variant_id)
        main = main.merge(emb, on="Name", how="left")
    return main

def featurize_sequence(seq, L_max=319):
    seq = str(seq).upper()
    onehot = np.zeros((L_max, len(AA)))
    for i, aa in enumerate(seq[:L_max]):
        if aa in AA_TO_IDX:
            onehot[i, AA_TO_IDX[aa]] = 1
    return onehot

def build_adj(L, window=6):
    A = np.zeros((L, L))
    for i in range(L):
        for j in range(max(0, i - window), min(L, i + window + 1)):
            A[i, j] = 1
    np.fill_diagonal(A, 1)
    return A

def create_splits(df, group_col="parent_id", train_frac=0.7):
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=42)
    groups = df[group_col] if group_col in df.columns else df.index
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    return train_df, test_df
