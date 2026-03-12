# PepCABO

Implementation for the paper **PepCABO: Latent-space Bayesian optimization for peptide–MHC binding using contrastive alignment**.

This repository provides a framework for optimizing peptides for MHC class I alleles using latent-space Bayesian optimization and baseline strategies.

## Supported objectives

- `BA` — Binding affinity
- `PS` — Presentation score


## Installation

Create the enviroment then activate it:

```bash
conda env create -f environment.yml
conda activate pepcabo
```

Download MHCflurry models:

```bash
mhcflurry-downloads fetch
```

Login to `wandb` for logging during optimization:

```bash
wandb login
```
Otherwise, set `TRACK_WITH_WANDB` in `exec.py` to `False`.


## Experiments

### Data and Pretrained Models

Pretrained models and allele embeddings, along with the datasets, are available [here](https://drive.google.com/file/d/1b5dcgxqkABXMYx4o14PNG3bqI4YdBDu1/view?usp=sharing).

Download the zip file and extract it in the root directory.

### Pretraining

Pretraining includes:

- Peptide VAE
- Allele VAE
- Latent-space alignment and GP surrogate training

To reproduce the pretrained models used in the paper, go to `/pepcabo/utils/pep_utils/seq_vae` and run the training scripts there using the provided examples.

### Optimization

Optimization experiments are run with `exec.py`.

Supported optimization strategies:

- Guided PepCABO
- PepCABO without guided initialization
- InvBO
- Vanilla LSBO



## Running optimization

General command:

```bash
python exec.py --allele <ALLELE> --seed <SEED> --objective <BA|PS> --init_strategy <STRATEGY> --bsz <BATCH_SIZE> --task <TASK_NAME> --output <OUTPUT_DIRECTORY>
```

### Arguments

| Argument | Description |
|---|---|
| `--allele` | Target MHC allele, for example `HLA-B*40:02` |
| `--seed` | Random seed |
| `--objective` | Optimization objective: `BA` or `PS` |
| `--init_strategy` | Initialization strategy |
| `--high` | Run the high-budget setting from the paper. If omitted, the low-budget setting is used |
| `--bsz` | Batch size |
| `--force` | Enforce evaluation of exactly `bsz` new peptides at each step |
| `--no-pretrain` | Disable pretrained models, GP, and aligned latent space |
| `--vanilla` | Use vanilla LSBO instead of InvBO |
| `--task` | Name of the experiment |
| `--output` | Directory to save the output dataframe |

### Examples

Guided PepCABO, low-budget:

```bash
python exec.py --allele HLA-B*40:02 --objective BA --init_strategy guided --bsz 20 --force --task BA_guided_pepcabo_low --output results
```

PepCABO without guided initialization, high-budget:

```bash
python exec.py --allele HLA-B*40:02 --objective BA --init_strategy random --bsz 5 --high --task BA_random_pepcabo_high --output results
```


## Notes

The repository also includes auxiliary notebooks for:

- Embedding generation
- Initialization analysis

## Citation

If you use this code, please cite the corresponding paper.

## Acknowledgment

This repository is a modified fork of [InvBO](https://github.com/mlvlab/InvBO).
