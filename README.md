# RedEx

Predicting experimental redox potentials for aqueous organic molecules using GNNs with transfer learning. 

RedEx is a GAT ensemble, pre-trained on computational DFT-calculated redox potentials and fine-tuned on experimental redox potentials curated from the literature. 

## Installation

```bash
conda env create -f environment.yml
conda activate redex
pip install -e .
```

## Pre-training

Pre-train on the computational RedoxDB dataset:

```bash
python scripts/pretrain.py --model gat        # single model
python scripts/pretrain.py --all              # all five architectures
```

Configs are in `configs/pretrain_{model}.yaml`. Checkpoints are saved to `weights/pretrained/`.

## Fine-tuning

Fine-tune on experimental data:

```bash
python scripts/finetune.py \
    --model gat \
    --pretrained weights/pretrained/gat_full_pretrain.pth \
    --config configs/finetune_gat.yaml
```

## Prediction

Pre-trained and fine-tuned weights are included and you can use RedEx to predict redox potentials with the below code.

```bash
python scripts/predict.py \
    --model-dir weights/finetuned/gat_partial \
    --smiles "OC1=CC=CC2=NC3=CC=CC=C3N=C21"
```

Or from a CSV (needs an `oxidised_smiles` column):

```bash
python scripts/predict.py \
    --model-dir weights/finetuned/gat_partial \
    --csv data/raw/Experimental_redox_potentials.csv \
    --output predictions.csv
```

## How to cite

*To be added upon publication.*

## Licence

MIT
