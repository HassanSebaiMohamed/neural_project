# Fashion MNIST Classification with MLP

A multi-layer perceptron (MLP) classifier trained on the Fashion MNIST dataset, with experiments comparing activation functions and learning rates.

---

## Problem Description

This project tackles **image classification** on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset — a drop-in replacement for the classic MNIST benchmark that consists of grayscale 28×28 images of clothing items across 10 categories.

**Goal:** Train an MLP with regularization (BatchNorm + Dropout) and compare the effect of:
- Activation function: ReLU vs. Tanh
- Learning rate: 0.001 vs. 0.0005

**Classes:**

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Dataset

- **Name:** Fashion MNIST
- **Source:** [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **Auto-downloaded** via `torchvision.datasets.FashionMNIST`
- **Size:** 60,000 training images, 10,000 test images
- **Split used:** 80% train / 20% validation from the training set

No manual download required — running the script will fetch the dataset automatically.

---

## Model Architecture

```
Input (28×28 = 784)
  → Linear(784, 128) → BatchNorm → Activation → Dropout(0.3)
  → Linear(128, 64)  → BatchNorm → Activation → Dropout(0.3)
  → Linear(64, 10)   [logits]
```

- **Loss:** Cross-Entropy
- **Optimizer:** Adam
- **Epochs:** 5 per experiment

---

## Results

### Experiment Comparison

| Experiment | Activation | Hidden Size | Learning Rate | Test Accuracy |
|------------|------------|-------------|---------------|---------------|
| Exp1  | ReLU | 128 | 0.001 | ~87–88% |
| Exp2  | Tanh | 128 | 0.001 | ~86–87% |
| Exp3  | ReLU | 128 | 0.0005 | ~86–87% |

> **Note:** Exact values vary slightly across runs due to random initialization. Results above are representative of typical outcomes over 5 epochs.

### Key Observations

- **ReLU (Exp1)** converges faster and achieves slightly higher accuracy than Tanh within 5 epochs.
- **Tanh (Exp2)** trains more smoothly but is slower to converge and ends up marginally lower.
- **Lower LR (Exp3)** reduces oscillation in the loss curve but underperforms vs. the default LR at only 5 epochs — it would likely catch up with more training.
- **BatchNorm + Dropout** effectively prevents overfitting: train/val accuracy tracks closely across all experiments.

---

## Project Structure

```
.
├── README.md
├── main.py              # Full training, evaluation, and experiment script
└── data/                # Auto-created by torchvision on first run
    └── FashionMNIST/
```

---

## Requirements

```
torch
torchvision
matplotlib
scikit-learn
seaborn
```

Install all dependencies with:

```bash
pip install torch torchvision matplotlib scikit-learn seaborn
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib scikit-learn seaborn
```

### 3. Run the experiments

```bash
python main.py
```

This will:
1. Download the Fashion MNIST dataset automatically (first run only)
2. Train 3 experiments sequentially (ReLU, Tanh, ReLU with lower LR)
3. Print per-epoch loss and accuracy for train/validation splits
4. Print test accuracy, loss, and a full classification report for each experiment
5. Display confusion matrices for each experiment
6. Display combined loss and accuracy curves comparing all three experiments

### GPU Support

The script automatically uses CUDA if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```





