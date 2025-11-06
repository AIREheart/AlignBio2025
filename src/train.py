"""
Training script for PETase Predictor.
- Loads MASTER_DB.xlsx
- Optionally loads embeddings from .npz or .npy via --emb_np and --emb_map_csv
- Trains FusionModel and saves best checkpoint by validation NDCG (primary)
"""

import argparse, os, json, math, time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import torch
from torch.utils.data import Dataset, DataLoader
from data_processing import detect_seq_column, featurize_seq_onehot, build_local_adj, create_grouped_splits, ndcg_at_k

from models import FusionModel

# ---------------------
# Dataset wrapper
# ---------------------
class PETaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, L_max: int, emb_map: dict = None):
        self.df = df.reset_index(drop=True)
        self.L = L_max
        self.emb_map = emb_map
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        seq = r["__seq"]
        seq_onehot = featurize_seq_onehot(seq, self.L)  # [L, 21]
        adj = build_local_adj(self.L)                   # [L, L]
        # node features: here we use one-hot as node features; if per-residue embeddings exist you can concat them
        node_feats = seq_onehot.copy()
        # global vector: use provided pooled embedding if exists in df (column emb_vec) else zeros
        glob = np.array(r["emb_vec"], dtype=np.float32) if "emb_vec" in r.index and not pd.isna(r["emb_vec"]) else np.zeros((args.glob_dim,), dtype=np.float32)
        y = np.array([r["_z_Tm"]], dtype=np.float32)  # single target; expand for multi-task
        mask = ~np.isnan(y)
        return {
            "seq_onehot": seq_onehot.astype(np.float32),
            "node_feats": node_feats.astype(np.float32),
            "adj": adj.astype(np.float32),
            "glob": glob.astype(np.float32),
            "y": y.astype(np.float32),
            "mask": mask.astype(np.float32)
        }

# ---------------------
# Training utilities
# ---------------------
def collate_fn(batch):
    return {
        "seq_onehot": torch.tensor(np.stack([b["seq_onehot"] for b in batch]), dtype=torch.float32),
        "node_feats": torch.tensor(np.stack([b["node_feats"] for b in batch]), dtype=torch.float32),
        "adj": torch.tensor(np.stack([b["adj"] for b in batch]), dtype=torch.float32),
        "glob": torch.tensor(np.stack([b["glob"] for b in batch]), dtype=torch.float32),
        "y": torch.tensor(np.stack([b["y"] for b in batch]), dtype=torch.float32),
        "mask": torch.tensor(np.stack([b["mask"] for b in batch]), dtype=torch.float32)
    }

def masked_mse(pred, targ, mask):
    diff = (pred - targ) * mask
    return (diff**2).sum() / (mask.sum() + 1e-8)

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/MASTER_DB.xlsx")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--emb_np", default=None, help="path to embeddings .npz (ids + pooled) OR .npy")
    parser.add_argument("--emb_map_csv", default=None, help="optional csv mapping ids->row index for emb_np")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--L_max", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Load workbook
    x = pd.read_excel(args.data, sheet_name=None)
    main_sheet = args.sheet if args.sheet is not None else max(x.keys(), key=lambda k: len(x[k]))
    df = x[main_sheet].copy()
    # detect seq col
    seq_col = detect_seq_column(df)
    if seq_col is None and "__seq" not in df.columns:
        raise RuntimeError("No sequence column detected.")
    df["__seq"] = df[seq_col].astype(str).str.upper().str.replace(r"\s+","", regex=True)
    # detect Tm
    tm_col = None
    for c in df.columns:
        if 'tm' in str(c).lower():
            tm_col = c; break
    if tm_col is None:
        raise RuntimeError("No Tm column detected.")
    df[tm_col] = pd.to_numeric(df[tm_col], errors='coerce')
    # Optional: load embeddings
    if args.emb_np:
        emb = np.load(args.emb_np, allow_pickle=True)
        # Expect emb to either be array [N, D] or dict-like with 'ids' & 'pooled'
        if isinstance(emb, np.lib.npyio.NpzFile) or isinstance(emb, dict):
            ids = emb['ids'] if 'ids' in emb else emb['ids']
            pooled = emb['pooled'] if 'pooled' in emb else emb['vecs']
            emb_map = {str(id_): i for i, id_ in enumerate(ids)}
            # attach pooled embeddings aligned by id column
            id_col = "Name" if "Name" in df.columns else df.columns[0]
            pooled_list = []
            for idx, row in df.iterrows():
                key = str(row[id_col])
                if key in emb_map:
                    pooled_list.append(pooled[emb_map[key]].astype(np.float32))
                else:
                    pooled_list.append(np.zeros(pooled.shape[1], dtype=np.float32))
            df["emb_vec"] = pooled_list
            glob_dim = pooled.shape[1]
        else:
            # raw array: assume same order
            arr = np.array(emb)
            df["emb_vec"] = list(arr.astype(np.float32))
            glob_dim = arr.shape[1]
    else:
        df["emb_vec"] = list([np.zeros(64, dtype=np.float32) for _ in range(len(df))])
        glob_dim = 64

    # Filter numeric Tm rows
    df = df[df[tm_col].notna()].reset_index(drop=True)
    # Create grouped splits
    group_col = "parent" if "parent" in df.columns else ("Name" if "Name" in df.columns else None)
    df = create_grouped_splits(df, group_col=group_col)
    # Standardize Tm on training set
    scaler = StandardScaler()
    train_vals = df.loc[df["_split"]=="train", tm_col].values.reshape(-1,1)
    scaler.fit(train_vals)
    df["_z_Tm"] = df[tm_col].apply(lambda x: float(scaler.transform([[x]])[0][0]) if pd.notna(x) else np.nan)

    # Build datasets
    L = int(min(args.L_max, int(df["__seq"].str.len().max())))
    train_df = df[df["_split"]=="train"].reset_index(drop=True)
    val_df = df[df["_split"]=="val"].reset_index(drop=True)
    test_df = df[df["_split"]=="test"].reset_index(drop=True)

    train_ds = PETaseDataset(train_df, L, None)
    val_ds = PETaseDataset(val_df, L, None)
    test_ds = PETaseDataset(test_df, L, None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = FusionModel(seq_channels=len(AA_IDX)+1, node_feat_dim=len(AA_IDX)+1, global_dim=glob_dim, out_tasks=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    best_val_ndcg = -1.0
    patience = 10
    no_improve = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            optimizer.zero_grad()
            seq = batch["seq_onehot"].to(device)
            node_feats = batch["node_feats"].to(device)
            adj = batch["adj"].to(device)
            glob = batch["glob"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            out = model(seq, node_feats, adj, glob)
            loss = masked_mse(out, y, mask)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * seq.shape[0]
            n += seq.shape[0]
        train_loss = total_loss / max(1, n)

        # validation: compute NDCG on pooled predictions
        model.eval()
        all_true = []
        all_pred = []
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq_onehot"].to(device)
                node_feats = batch["node_feats"].to(device)
                adj = batch["adj"].to(device)
                glob = batch["glob"].to(device)
                y = batch["y"].cpu().numpy()
                mask = batch["mask"].numpy().astype(bool)
                pred = model(seq, node_feats, adj, glob).cpu().numpy()
                # inverse-transform
                valid = mask[:,0]
                if valid.sum()==0:
                    continue
                true_vals = scaler.inverse_transform(y[valid,0].reshape(-1,1)).flatten()
                pred_vals = scaler.inverse_transform(pred[valid,0].reshape(-1,1)).flatten()
                all_true.extend(true_vals.tolist()); all_pred.extend(pred_vals.tolist())
        # compute val ndcg & spearman
        if len(all_true) > 1:
            val_ndcg = ndcg_at_k(np.array(all_true), np.array(all_pred), k=min(10, len(all_true)))
            val_spearman = spearmanr(np.array(all_true), np.array(all_pred)).correlation
        else:
            val_ndcg = 0.0; val_spearman = None

        print(f"[Epoch {epoch}] train_loss={train_loss:.5f} val_ndcg@10={val_ndcg:.4f} val_spearman={val_spearman}")

        # checkpointing by val_ndcg
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "scaler": scaler,
                "meta": {"L": L, "glob_dim": glob_dim, "epoch": epoch}
            }, os.path.join(args.out_dir, "best.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print("No improvement; early stopping.")
                break

    # Finally evaluate on test set using best checkpoint
    ck = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    all_true = []; all_pred = []
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq_onehot"].to(device)
            node_feats = batch["node_feats"].to(device)
            adj = batch["adj"].to(device)
            glob = batch["glob"].to(device)
            y = batch["y"].cpu().numpy()
            mask = batch["mask"].numpy().astype(bool)
            pred = model(seq, node_feats, adj, glob).cpu().numpy()
            valid = mask[:,0]
            if valid.sum()==0:
                continue
            true_vals = scaler.inverse_transform(y[valid,0].reshape(-1,1)).flatten()
            pred_vals = scaler.inverse_transform(pred[valid,0].reshape(-1,1)).flatten()
            all_true.extend(true_vals.tolist()); all_pred.extend(pred_vals.tolist())

    if len(all_true) > 1:
        test_ndcg = ndcg_at_k(np.array(all_true), np.array(all_pred), k=min(10, len(all_true)))
        test_spearman = spearmanr(np.array(all_true), np.array(all_pred)).correlation
    else:
        test_ndcg = None; test_spearman = None

    metrics = {"test_ndcg@10": test_ndcg, "test_spearman": float(test_spearman) if test_spearman is not None else None}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    print("Training finished. Metrics:", metrics)
