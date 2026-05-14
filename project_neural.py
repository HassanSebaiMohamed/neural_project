import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATA (Augmentation + Normalization)
# =========================
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =========================
# MODEL (MLP + Dropout)
# =========================
class MLP(nn.Module):
    def __init__(self, activation="relu", hidden_size=128):
        super(MLP, self).__init__()

        if activation == "relu":
            act = nn.ReLU()
        else:
            act = nn.Tanh()

        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28*28, hidden_size),
            act,
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size // 2),
            act,
            nn.Dropout(0.3),

            nn.Linear(hidden_size // 2, 10)
        )

    def forward(self, x):
        return self.model(x)

# =========================
# TRAINING FUNCTION
# =========================
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()

    losses = []
    accuracies = []

    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        losses.append(running_loss)
        accuracies.append(acc)

        print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Accuracy={acc:.2f}%")

    return losses, accuracies

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, test_loader, criterion):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Loss: {total_loss:.4f}")

    return acc, total_loss

# =========================
# EXPERIMENTS
# =========================
results = []

def run_experiment(name, activation, hidden_size, lr):

    print("\n====================")
    print(name)
    print("====================")

    model = MLP(activation, hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss, acc = train_model(model, train_loader, criterion, optimizer, epochs=5)

    test_acc, test_loss = evaluate(model, test_loader, criterion)

    results.append({
        "name": name,
        "acc": test_acc,
        "loss": test_loss
    })

    return loss, acc

# =========================
# RUN EXPERIMENTS
# =========================
loss1, acc1 = run_experiment(
    "Exp1_ReLU_128_lr0.001",
    "relu",
    128,
    0.001
)

loss2, acc2 = run_experiment(
    "Exp2_Tanh_256_lr0.0005",
    "tanh",
    256,
    0.0005
)

# =========================
# VISUALIZATION
# =========================
plt.figure()
plt.plot(loss1, label="Experiment 1 Loss")
plt.plot(loss2, label="Experiment 2 Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(acc1, label="Experiment 1 Accuracy")
plt.plot(acc2, label="Experiment 2 Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.show()

# =========================
# FINAL RESULTS
# =========================
print("\n===== FINAL COMPARISON =====")
for r in results:
    print(r)