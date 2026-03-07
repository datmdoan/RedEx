# RedEx

Code and pre-trained weights for:

> **Transfer Learning with Graph Neural Networks for Molecular Redox Potential Prediction**
>
> *Citation to be added upon publication.*

Five GNN architectures (GCN, GAT, GIN, GraphSAGE, D-MPNN) pre-trained on ~7,500 DFT redox potentials from RedoxDB, then fine-tuned on 264 experimental measurements. The best model (GAT, partial freeze) ships as a 5-fold ensemble ready to use.

---

## Installation

Install PyTorch and PyG first to match your CUDA version — see [pytorch.org](https://pytorch.org) and [pyg.org](https://pyg.org).

```bash
conda create -n redex python=3.11 -y && conda activate redex

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric

pip install -r requirements.txt
pip install -e .
```

---

## Predict

```bash
python scripts/predict.py \
    --model-dir weights/finetuned/gat_partial \
    --smiles "OC1=CC=CC2=NC3=CC=CC=C3N=C21"
```

From a CSV (needs an `oxidised_smiles` column):

```bash
python scripts/predict.py \
    --model-dir weights/finetuned/gat_partial \
    --csv data/raw/Experimental_redox_potentials.csv \
    --output predictions.csv
```

---

## Pre-train from scratch

Each model uses its own HPO-tuned config (`configs/pretrain_{model}.yaml`).

```bash
python scripts/pretrain.py --model gat        # single model
python scripts/pretrain.py --all              # all five
```

Checkpoints are saved to `weights/pretrained/`.

---

## Fine-tune

```bash
python scripts/finetune.py \
    --model gat \
    --pretrained weights/pretrained/gat_full_pretrain.pth \
    --config configs/finetune_gat.yaml
```

Four freeze strategies are available via the `freeze` key in the config: `none`, `head_only`, `readout_head`, or `top_layers`. The `freeze_ratio` parameter controls what fraction of the convolutional layers are frozen from the bottom.

---

## Data

- `data/raw/RedoxDB_Complete.csv` — computational dataset used for pre-training
- `data/raw/Experimental_redox_potentials.csv` — experimental dataset used for fine-tuning

---

## Licence

MIT — see [LICENSE](LICENSE).


```
RedEx/
├── RedEx/                      # installable Python package
│   ├── models/                 # GNN architectures + readout layers
│   ├── data/                   # featuriser, dataset loading, transforms
│   └── utils/                  # training helpers, NN building blocks
├── scripts/
│   ├── pretrain.py             # pre-train on computational data
│   ├── finetune.py             # fine-tune on experimental data (5-fold CV)
│   └── predict.py              # predict from SMILES using a saved ensemble
├── configs/
│   ├── pretrain_{model}.yaml   # HPO-optimised pre-training settings (×5)
│   └── finetune_gat.yaml       # optimised GAT fine-tuning settings
├── weights/
│   ├── pretrained/             # 5 pre-trained checkpoints
│   └── finetuned/gat_partial/  # GAT ensemble (5 folds + metadata)
├── data/raw/                   # computational & experimental datasets
├── pyproject.toml              # packaging & dependency metadata
├── requirements.txt            # pinned dependency versions
└── LICENSE                     # MIT licence
```

## Installation

Requires Python ≥ 3.10.  PyTorch and PyG should be installed first to match your
CUDA version — see https://pytorch.org and https://pyg.org for instructions.

```bash
conda create -n redex python=3.11 -y && conda activate redex

# install PyTorch (adjust cu121 to match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# install PyG
pip install torch-geometric

# install remaining dependencies and the package itself
pip install -r requirements.txt
pip install -e .
```

## Usage

### Predict redox potentials

The shipped GAT ensemble can be used directly on arbitrary SMILES:

```bash
python scripts/predict.py \
    --model-dir weights/finetuned/gat_partial \
    --smiles "OC1=CC=CC2=NC3=CC=CC=C3N=C21" "O=C1C=CC=C2C=C3C(C=C21)=CC=CC3=O"
```

For batch prediction from a CSV file (must contain an `oxidised_smiles` column):

```bash
python scripts/predict.py \
    --model-dir weights/finetuned/gat_partial \
    --csv data/raw/Experimental_redox_potentials.csv \
    --output predictions.csv
```

### Pre-train from scratch

Train on the computational RedoxDB dataset with a 90/10 train/validation split.
Each model has its own HPO-tuned config in `configs/pretrain_{model}.yaml`.

```bash
# single model (automatically uses configs/pretrain_gat.yaml)
python scripts/pretrain.py --model gat

# all five models, each using its own config
python scripts/pretrain.py --all

# override the config explicitly
python scripts/pretrain.py --model gat --config configs/pretrain_gat.yaml
```

Checkpoints are written to `weights/pretrained/`.

### Fine-tune on experimental data

Fine-tune a pre-trained checkpoint with 5-fold stratified group CV.  Hyperparameters
are read from a YAML config; any CLI argument overrides the corresponding config value.

```bash
python scripts/finetune.py \
    --model gat \
    --pretrained weights/pretrained/gat_full_pretrain.pth \
    --config configs/finetune_gat.yaml
```

#### Freeze strategies

| Strategy | What is trained |
|---|---|
| `none` | All layers (full fine-tuning) |
| `head_only` | Prediction head only |
| `readout_head` | Readout pooling + prediction head |
| `top_layers` | Top (1 − *R*) fraction of conv layers, readout, and head |

The `freeze_ratio` parameter in the config controls *R*, the proportion of
convolutional layers frozen from the bottom of the network.

## Data

- **Computational (RedoxDB)** — 7,575 DFT-computed redox potentials.
- **Experimental** — 264 experimental redox potentials (cyclic voltammetry measurements).
  split into CV and held-out test sets using group-stratified sampling.

Molecules are featurised with a 74-dimensional atom descriptor and a 13-dimensional
bond descriptor via RDKit (see `RedEx/data/featuriser.py`).

## Licence

MIT — see [LICENSE](LICENSE).
