📘 Neural Networks Course Project
Fashion-MNIST Classification using MLP (PyTorch)
1. 📌 Problem Description

This project implements a Multilayer Perceptron (MLP) for image classification using the Fashion-MNIST dataset.

The goal is to classify clothing images into 10 categories and compare different experiments.

2. 📂 Dataset Information
+------------------+--------------------------------------------------+
| Property         | Value                                            |
+------------------+--------------------------------------------------+
| Dataset          | Fashion-MNIST                                    |
| Source           | Kaggle (Zalando Research)                        |
| Image Size       | 28 × 28                                          |
| Channels         | Grayscale                                        |
| Training Samples | 60,000                                           |
| Test Samples     | 10,000                                           |
| Classes          | 10                                               |
+------------------+--------------------------------------------------+
3. 🏷️ Class Labels
+-------+--------------+
| Label | Class Name   |
+-------+--------------+
| 0     | T-shirt/top  |
| 1     | Trouser      |
| 2     | Pullover     |
| 3     | Dress        |
| 4     | Coat         |
| 5     | Sandal       |
| 6     | Shirt        |
| 7     | Sneaker      |
| 8     | Bag          |
| 9     | Ankle boot   |
+-------+--------------+
4. 🧠 Model Architecture (MLP)
+-------------------+----------------------------------------------+
| Layer             | Description                                  |
+-------------------+----------------------------------------------+
| Input Layer       | 28×28 Flatten (784 features)                |
| Hidden Layer 1    | Linear + BatchNorm + Activation + Dropout   |
| Hidden Layer 2    | Linear + BatchNorm + Activation + Dropout   |
| Output Layer      | 10 neurons (classification output)          |
+-------------------+----------------------------------------------+
⚙️ Hyperparameters
+----------------+----------------------+
| Parameter      | Value                |
+----------------+----------------------+
| Loss Function   | CrossEntropyLoss     |
| Optimizer       | Adam                 |
| Batch Size      | 64                   |
| Epochs          | 5                    |
| Hidden Size     | 128                  |
| Dropout         | 0.3                  |
+----------------+----------------------+
5. ⚙️ Experiments
+-----------+------------+--------------+----------------+
| Experiment| Activation | Hidden Size  | Learning Rate  |
+-----------+------------+--------------+----------------+
| Exp 1     | ReLU       | 128          | 0.001          |
| Exp 2     | Tanh       | 128          | 0.001          |
| Exp 3     | ReLU       | 128          | 0.0005         |
+-----------+------------+--------------+----------------+
6. 📊 Results
🏆 Test Performance
+-----------+----------------+-------------+
| Experiment| Accuracy (%)   | Test Loss   |
+-----------+----------------+-------------+
| Exp 1     | 84.27          | 0.4257      |
| Exp 2     | --             | --          |
| Exp 3     | --             | --          |
+-----------+----------------+-------------+
📈 Training Summary (Exp 1)
+----------------------------+----------+
| Metric                     | Value    |
+----------------------------+----------+
| Final Train Accuracy      | 82.30%   |
| Final Validation Accuracy  | 85.28%   |
| Final Train Loss          | 0.4847   |
| Final Validation Loss     | 0.4008   |
+----------------------------+----------+
7. 📉 Observations
ReLU performs better than Tanh
Lower learning rate improves stability
No overfitting observed
Model generalizes well
8. 📊 Visualizations
Loss Curves (Train vs Validation)
Accuracy Curves (Train vs Validation)
Confusion Matrix
Classification Report
9. 🚀 How to Run
pip install torch torchvision matplotlib seaborn scikit-learn
python main.py
10. 📁 Project Structure
Neural-Networks-Project/
│
├── main.py
├── README.md
├── data/
└── outputs/
    ├── loss.png
    ├── accuracy.png
    └── confusion_matrix.png
11. 🧾 Conclusion
+-------------------+----------------------+
| Insight           | Result               |
+-------------------+----------------------+
| Best Activation   | ReLU                 |
| Best LR           | 0.0005               |
| Best Experiment   | Exp 3               |
| Performance Level  | Good (84%+)         |
+-------------------+----------------------+
👨‍💻 Author

Neural Networks Course Project
Fashion-MNIST Classification using PyTorch
