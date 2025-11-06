import argparse, torch, numpy as np, pandas as pd
from data_processing import load_data, featurize_sequence, build_adj, create_splits
from models import FusionModel
from sklearn.preprocessing import StandardScaler

def run_training(data_path, epochs=20, device="cpu"):
    df = load_data(data_path)
    df = df.dropna(subset=["Tm"])
    L_max = 319
    seqs = torch.tensor(np.stack([featurize_sequence(s, L_max) for s in df["sequence"]]))
    adjs = torch.tensor(np.stack([build_adj(L_max) for _ in range(len(df))]))
    emb_cols = [c for c in df.columns if "emb" in c.lower() or str(df[c].dtype).startswith("float")]
    Xglob = torch.tensor(df[emb_cols].fillna(0).to_numpy(), dtype=torch.float32)
    y = torch.tensor(df["Tm"].values, dtype=torch.float32).unsqueeze(1)
    scaler = StandardScaler().fit(y)
    y = torch.tensor(scaler.transform(y), dtype=torch.float32)

    model = FusionModel(in_channels=21, glob_dim=Xglob.shape[1], n_tasks=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(seqs.to(device), adjs.to(device), Xglob.to(device))
        loss = loss_fn(pred, y.to(device))
        loss.backward()
        opt.step()
        print(f"Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), "checkpoints/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/MASTER_DB.xlsx")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_training(args.data, args.epochs, args.device)
