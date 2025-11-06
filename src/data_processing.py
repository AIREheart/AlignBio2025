"""
Data loading and preprocessing utilities.

Functions:
- load_workbook(path)
- detect_columns(df)
- merge_embeddings(main_df, emb_df)
- create_grouped_splits(df, group_col, train_frac)
- build_datasets(df, L_max)
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


AA = list("ARNDCQEGHILKMFPSTWYV")


def load_workbook(path):
xls = pd.read_excel(path, sheet_name=None)
return xls



# For brevity this file contains utility functions used by train.py
