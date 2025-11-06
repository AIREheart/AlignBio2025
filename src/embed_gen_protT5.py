"""
Generate ProtT5 embeddings (per-sequence pooled vector or per-residue arrays).
Outputs a .npz file with arrays:
 - ids: list of variant IDs
 - seqs: list of sequences
 - pooled: [N, D] global vectors (mean pooling)
 - residues: [N, L, D_res] optional per-residue embeddings (if memory allows)
"""
import argparse, os, json
import numpy as np, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

def load_sheet(xlsx_path, sheet='150petase', id_col='Name', seq_col=None):
    xls = pd.read_excel(xlsx_path, sheet_name=None)
    df = xls[sheet].copy()
    if seq_col is None:
        # choose a likely sequence column
        for c in df.columns:
            sample = df[c].dropna().astype(str).iloc[:10].tolist() if df[c].dropna().shape[0]>0 else []
            if sample and all(len(s)>5 for s in sample):
                seq_col = c; break
    ids = df[id_col].astype(str).tolist()
    seqs = df[seq_col].astype(str).tolist()
    return ids, seqs

def main(args):
    # model id: Rostlab/prot_t5_xl_uniref50 (large); choose smaller if memory-limited
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(args.device)
    model.eval()

    ids, seqs = load_sheet(args.data, sheet=args.sheet, id_col=args.id_col, seq_col=args.seq_col)
    pooled_list = []
    residues_list = []
    for seq in tqdm(seqs, desc="Embedding"):
        # ProtT5 expects spaces between AA tokens; transform sequence into space-separated
        seq_proc = " ".join(list(seq.replace("U","X").replace("Z","X").replace("O","X")))
        inputs = tokenizer(seq_proc, return_tensors='pt')
        with torch.no_grad():
            out = model(**{k: v.to(args.device) for k,v in inputs.items()})
            # last_hidden_state: [1, L, D]
            token_embs = out.last_hidden_state.squeeze(0).cpu().numpy()
            # mean pooling excluding special tokens:
            pooled = token_embs.mean(axis=0)
            pooled_list.append(pooled.astype(np.float32))
            # if requested: per-residue
            if args.per_residue:
                residues_list.append(token_embs.astype(np.float32))
    pooled = np.stack(pooled_list, axis=0)
    out = {"ids": ids, "pooled": pooled}
    if args.per_residue:
        # residues_list can be ragged; save as object array or pad to L_max
        out["residues"] = residues_list
    np.savez_compressed(args.out, **out)
    print("Saved embeddings to:", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="MASTER_DB.xlsx")
    parser.add_argument("--sheet", default="150petase")
    parser.add_argument("--id_col", default="Name")
    parser.add_argument("--seq_col", default=None)
    parser.add_argument("--out", default="data/petase_prott5.npz")
    parser.add_argument("--model", default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per_residue", action="store_true")
    args = parser.parse_args()
    main(args)
