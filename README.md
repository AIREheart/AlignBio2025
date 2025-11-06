# AlignBio2025

Current approach: training a hybrid CNN>GNN model for predicting properties of PETase variants.

Quickstart (in dev)
1 - create venv & install requirements:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2 - EDA & preprocess data + conduct demo training?

3 - training with GPU 
```
python3 src/train.py --epochs 100 --batch_size 32 --device cuda
```

4 - evaluate 
```
python3 src/evaluate.py --model 
```
