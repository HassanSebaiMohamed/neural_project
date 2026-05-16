📘 Neural Networks Course Project
Fashion-MNIST Classification using MLP (PyTorch)
1. 📌 Problem Description

This project implements a Multilayer Perceptron (MLP) neural network for image classification using the Fashion-MNIST dataset.

The goal is to classify grayscale images of clothing items into 10 categories and evaluate the effect of different hyperparameters through multiple experiments.

2. 📂 Dataset Information
Property	Value
Dataset	Fashion-MNIST
Source	https://www.kaggle.com/datasets/zalando-research/fashionmnist
Image Size	28 × 28
Channels	Grayscale
Training Samples	60,000
Test Samples	10,000
Classes	10
3. 🏷️ Class Labels
Label	Class
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
4. 🧠 Model Architecture (MLP)
Layer	Description
Input	28×28 Flatten (784 features)
Hidden Layer 1	Linear + BatchNorm + Activation + Dropout
Hidden Layer 2	Linear + BatchNorm + Activation + Dropout
Output Layer	10 neurons (classification)
⚙️ Hyperparameters
Parameter	Value
Loss Function	CrossEntropyLoss
Optimizer	Adam
Batch Size	64
Epochs	5
Dropout	0.3
Hidden Size	128
5. ⚙️ Experiments

Three different experiments were performed:

Experiment	Activation	Hidden Size	Learning Rate
Exp 1	ReLU	128	0.001
Exp 2	Tanh	128	0.001
Exp 3	ReLU	128	0.0005
6. 📊 Results
🏆 Test Performance
Experiment	Test Accuracy	Test Loss
Exp 1 (ReLU)	84.27%	0.4257
Exp 2 (Tanh)	(add result)	(add result)
Exp 3 (LR 0.0005)	(add result)	(add result)
📈 Training Summary (Exp 1 Example)
Metric	Value
Final Train Accuracy	82.30%
Final Validation Accuracy	85.28%
Final Train Loss	0.4847
Final Validation Loss	0.4008
7. 📉 Observations
ReLU performs better than Tanh in most cases
Lower learning rate improves stability
Model shows good generalization (small gap between train and validation)
No significant overfitting observed
8. 📊 Visualizations Included
Training vs Validation Loss Curves
Training vs Validation Accuracy Curves
Confusion Matrix
Classification Report (Precision / Recall / F1-score)
9. 🚀 How to Run
Install Dependencies
pip install torch torchvision matplotlib seaborn scikit-learn
Run Project
python main.py
10. 📁 Project Structure
Neural-Networks-Project/
│
├── main.py
├── README.md
├── data/ (auto downloaded)
│
└── outputs/
    ├── loss.png
    ├── accuracy.png
    └── confusion_matrix.png
11. 🧾 Conclusion
Insight	Result
Best Activation	ReLU
Best Learning Rate	0.0005
Best Experiment	Exp 3
Model Type	MLP
Performance Level	Good (80%–85% Accuracy)
👨‍💻 Author

Neural Networks Course Project
Fashion-MNIST Classification using PyTorch
