"""
Evaluation script to compute Spearman and NDCG on a dataset using a saved checkpoint.
"""
import argparse, json, os
import numpy as np, pandas as pd
import torch
from scipy.stats import spearmanr
from data_processing import detect_seq_column, build_local_adj, featurize_seq_onehot, ndcg_at_k

from models import FusionModel

def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    return ck

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--emb_np", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Loading logic similar to train.py, reconstruct test loader, load model, compute metrics and print them.

if __name__ == "__main__":
    main()
