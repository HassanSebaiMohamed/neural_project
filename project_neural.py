import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATA
# =========================
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =========================
# MODEL
# =========================
class MLP(nn.Module):
    def __init__(self, activation="relu", hidden_size=128):
        super().__init__()

        act = nn.ReLU() if activation == "relu" else nn.Tanh()

        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28*28, hidden_size),
            nn.BatchNorm1d(hidden_size),
            act,
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            act,
            nn.Dropout(0.3),

            nn.Linear(hidden_size // 2, 10)
        )

    def forward(self, x):
        return self.model(x)

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):

        model.train()
        total, correct, running_loss = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # VALIDATION
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item()

                _, pred = torch.max(out, 1)
                val_total += y.size(0)
                val_correct += (pred == y).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: TL={train_loss:.4f} VL={val_loss:.4f} TA={train_acc:.2f}% VA={val_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs

# =========================
# EVALUATION (FULL FIXED)
# =========================
def evaluate(model, test_loader, criterion):

    model.eval()

    y_true, y_pred = [], []
    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()

            _, pred = torch.max(out, 1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

            total += y.size(0)
            correct += (pred == y).sum().item()

    acc = 100 * correct / total
    avg_loss = total_loss / len(test_loader)

    print("\n===== TEST ACCURACY =====")
    print(acc)

    print("\n===== TEST LOSS =====")
    print(avg_loss)

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10)
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return acc

# =========================
# EXPERIMENT FUNCTION
# =========================
def run_experiment(name, activation, hidden_size=128, lr=0.001):

    print("\n====================")
    print(name)
    print("====================")

    model = MLP(activation, hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs=5
    )

    evaluate(model, test_loader, criterion)

    return train_losses, val_losses, train_accs, val_accs

# =========================
# EXPERIMENTS
# =========================

t1, v1, ta1, va1 = run_experiment("Exp1_ReLU", "relu")
t2, v2, ta2, va2 = run_experiment("Exp2_Tanh", "tanh")
t3, v3, ta3, va3 = run_experiment("Exp3_ReLU_LR0005", "relu", 128, 0.0005)

# =========================
# VISUALIZATION
# =========================

plt.figure()
plt.plot(t1, label="Exp1 Train")
plt.plot(v1, label="Exp1 Val")
plt.plot(t2, label="Exp2 Train")
plt.plot(v2, label="Exp2 Val")
plt.plot(t3, label="Exp3 Train")
plt.plot(v3, label="Exp3 Val")
plt.title("Loss Curves")
plt.legend()
plt.show()

plt.figure()
plt.plot(ta1, label="Exp1 Train")
plt.plot(va1, label="Exp1 Val")
plt.plot(ta2, label="Exp2 Train")
plt.plot(va2, label="Exp2 Val")
plt.plot(ta3, label="Exp3 Train")
plt.plot(va3, label="Exp3 Val")
plt.title("Accuracy Curves")
plt.legend()
plt.show()
