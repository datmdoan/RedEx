# RedEx

Code and weights for:

> **Transfer Learning with Graph Neural Networks for Molecular Redox Potential Prediction**
>
> *Citation to be added upon publication.*

Five GNN architectures (GCN, GAT, GIN, GraphSAGE, D-MPNN) pre-trained on ~7,500 DFT redox potentials from RedoxDB, then fine-tuned on 264 experimental measurements. The best model (GAT) is shipped as a 5-fold ensemble you can run straight away.

## Installation

Python ≥ 3.10. Install PyTorch and PyG first to match your CUDA version
(see https://pytorch.org and https://pyg.org).

```bash
conda create -n redex python=3.11 -y && conda activate redex

# PyTorch — change cu121 to match your driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric

pip install -r requirements.txt
pip install -e .
```

## Quick start — predict from SMILES

```bash
# one or more SMILES
python scripts/predict.py \
    --model-dir weights/finetuned/gat_partial \
    --smiles "OC1=CC=CC2=NC3=CC=CC=C3N=C21"

# from a CSV (needs an oxidised_smiles column)
python scripts/predict.py \
    --model-dir weights/finetuned/gat_partial \
    --csv data/raw/Experimental_redox_potentials.csv \
    --output predictions.csv
```

## Re-training

### Pre-train on computational data

Each architecture has its own config with HPO-tuned hyperparameters.

```bash
python scripts/pretrain.py --model gat        # single model
python scripts/pretrain.py --all              # all five
```

Configs live in `configs/pretrain_{model}.yaml`. Checkpoints are saved to `weights/pretrained/`.

### Fine-tune on experimental data

```bash
python scripts/finetune.py \
    --model gat \
    --pretrained weights/pretrained/gat_full_pretrain.pth \
    --config configs/finetune_gat.yaml
```

Settings are read from the YAML config; any CLI flag overrides the config value.

Four freeze strategies are available: `none` (train everything), `head_only`,
`readout_head`, and `top_layers` (controlled by `freeze_ratio` — the fraction of
convolutional layers frozen from the bottom up).

## What's in the repo

| Path | Contents |
|---|---|
| `RedEx/` | Installable Python package — models, featuriser, training utilities |
| `scripts/` | `pretrain.py`, `finetune.py`, `predict.py` |
| `configs/` | YAML configs for pre-training (×5) and fine-tuning |
| `weights/pretrained/` | Pre-trained checkpoints for all five architectures |
| `weights/finetuned/gat_partial/` | Fine-tuned GAT ensemble (5 folds + metadata) |
| `data/raw/` | RedoxDB (computational) and experimental datasets |

## Licence

MIT — see [LICENSE](LICENSE).
