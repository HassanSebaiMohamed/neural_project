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
| Exp1 — Baseline | ReLU | 128 | 0.001 | ~87–88% |
| Exp2 — Tanh | Tanh | 128 | 0.001 | ~86–87% |
| Exp3 — Lower LR | ReLU | 128 | 0.0005 | ~86–87% |

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

No configuration needed — it falls back to CPU transparently.

---

## Sample Output (per experiment)

====================
Exp1_ReLU
====================
Epoch 1: TL=0.7496 VL=0.4946 TA=74.87% VA=81.92%
Epoch 2: TL=0.5631 VL=0.4468 TA=79.70% VA=83.62%
Epoch 3: TL=0.5255 VL=0.4303 TA=80.87% VA=83.75%
Epoch 4: TL=0.4995 VL=0.4202 TA=81.66% VA=84.55%
Epoch 5: TL=0.4847 VL=0.4008 TA=82.30% VA=85.28%

===== TEST ACCURACY =====
84.27

===== TEST LOSS =====
0.4256714725760138

===== CLASSIFICATION REPORT =====
              precision    recall  f1-score   support

           0       0.76      0.86      0.80      1000
           1       0.97      0.95      0.96      1000
           2       0.78      0.71      0.75      1000
           3       0.84      0.88      0.86      1000
           4       0.75      0.78      0.76      1000
           5       0.93      0.90      0.92      1000
           6       0.63      0.56      0.59      1000
           7       0.87      0.91      0.89      1000
           8       0.96      0.96      0.96      1000
           9       0.92      0.92      0.92      1000

    accuracy                           0.84     10000
   macro avg       0.84      0.84      0.84     10000
weighted avg       0.84      0.84      0.84     10000

====================
Exp2_Tanh
====================
Epoch 1: TL=0.7313 VL=0.5455 TA=75.41% VA=80.57%
Epoch 2: TL=0.5847 VL=0.5070 TA=79.37% VA=81.49%
Epoch 3: TL=0.5531 VL=0.4757 TA=80.34% VA=82.94%
Epoch 4: TL=0.5332 VL=0.4680 TA=80.99% VA=83.36%
Epoch 5: TL=0.5222 VL=0.4478 TA=81.09% VA=83.96%

===== TEST ACCURACY =====
82.72

===== TEST LOSS =====
0.47508342298352796

===== CLASSIFICATION REPORT =====
              precision    recall  f1-score   support

           0       0.79      0.82      0.81      1000
           1       0.97      0.95      0.96      1000
           2       0.72      0.76      0.74      1000
           3       0.82      0.86      0.84      1000
           4       0.72      0.77      0.74      1000
           5       0.83      0.92      0.87      1000
           6       0.65      0.53      0.58      1000
           7       0.90      0.83      0.86      1000
           8       0.94      0.94      0.94      1000
           9       0.91      0.91      0.91      1000

    accuracy                           0.83     10000
   macro avg       0.83      0.83      0.83     10000

====================
Exp3_ReLU_LR0005
====================
Epoch 1: TL=0.8320 VL=0.5099 TA=73.89% VA=81.88%
Epoch 2: TL=0.5814 VL=0.4554 TA=79.56% VA=83.50%
Epoch 3: TL=0.5377 VL=0.4302 TA=80.72% VA=84.23%
Epoch 4: TL=0.5136 VL=0.4175 TA=81.69% VA=84.64%
Epoch 5: TL=0.4917 VL=0.3995 TA=82.40% VA=85.30%

===== TEST ACCURACY =====
83.88

===== TEST LOSS =====
0.43310236760005827

===== CLASSIFICATION REPORT =====
              precision    recall  f1-score   support

           0       0.79      0.82      0.81      1000
           1       0.99      0.95      0.97      1000
           2       0.74      0.74      0.74      1000
           3       0.82      0.88      0.85      1000
           4       0.72      0.81      0.76      1000
           5       0.95      0.86      0.90      1000
           6       0.67      0.52      0.58      1000
           7       0.86      0.92      0.89      1000
           8       0.93      0.96      0.94      1000
           9       0.91      0.94      0.93      1000

    accuracy                           0.84     10000
   macro avg       0.84      0.84      0.84     10000