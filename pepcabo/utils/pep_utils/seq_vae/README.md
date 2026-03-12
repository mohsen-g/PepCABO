# Joint Peptide–Allele Model

Jointly trains **peptide and allele VAEs with a contrastive objective and a deep-kernel Gaussian Process surrogate** to model peptide–MHC binding or presentation scores.

## `--objective`

Specifies which target value the GP surrogate learns to predict.

Options:

- `BA` – MHCflurry **binding affinity**
- `PS` – MHCflurry **presentation score**
- `Experimental` – **experimental affinity values**

## Examples

Train using **binding affinity**:

```bash
python3 JointTraining.py --objective BA --con-loss 2 --trans-loss 2 --temp 1 --weighted-loss --hidden-size 64 16 --ystar 0.574 --batch-size 8192
```

Train using **presentation score**:

```bash
python3 JointTraining.py --objective PS --con-loss 2 --trans-loss 2 --temp 1 --weighted-loss --hidden-size 256 256 --ystar 0.9 --batch-size 8192
```

Train using **experimental measurements**:

```bash
python3 JointTraining.py --objective Experimental --con-loss 2 --trans-loss 2 --temp 1 --weighted-loss --hidden-size 64 16 --ystar 0.65 --batch-size 4096
```