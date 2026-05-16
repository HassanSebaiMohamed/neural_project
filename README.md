 Neural Networks Course Project
Fashion-MNIST Classification using MLP (PyTorch)
1.  Problem Description

This project focuses on implementing a Multilayer Perceptron (MLP) neural network to solve an image classification problem using the Fashion-MNIST dataset.

The objective is to classify grayscale images of clothing items into 10 different categories, such as:

T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot

The project also includes multiple experiments to analyze how different hyperparameters affect model performance.

2.  Dataset Information
Dataset Name: Fashion-MNIST
Source: https://www.kaggle.com/datasets/zalando-research/fashionmnist
Type: Image Classification Dataset
Image Size: 28 × 28 pixels (Grayscale)
Number of Classes: 10
Training Samples: 60,000
Test Samples: 10,000
3.  Model Architecture (MLP)

The implemented model is a Multilayer Perceptron (MLP) consisting of:

Input Layer: Flattened 28×28 image (784 features)
Hidden Layer 1: Fully Connected + Batch Normalization + Activation + Dropout
Hidden Layer 2: Fully Connected + Batch Normalization + Activation + Dropout
Output Layer: 10 neurons (one per class)
 Activation Functions Used:
ReLU
Tanh (for comparison experiments)
 Loss Function:
CrossEntropyLoss
 Optimizer:
Adam Optimizer
4.  Experimental Setup

To evaluate model performance, three different experiments were conducted:

Experiment	Activation	Hidden Size	Learning Rate
Exp 1	ReLU	128	0.001
Exp 2	Tanh	128	0.001
Exp 3	ReLU	128	0.0005
5.  Results & Evaluation

The model was evaluated using the following metrics:

Accuracy
Loss
Confusion Matrix
Precision / Recall / F1-score
 Performance Comparison
Experiment	Accuracy (%)	Observations
Exp 1	~82 - 88	Best overall performance using ReLU
Exp 2	~80 - 85	Slightly lower due to Tanh activation
Exp 3	~83 - 89	More stable training with lower learning rate
6.  Visualizations Included

The project includes the following visual outputs:

Training vs Validation Loss Curves
Training vs Validation Accuracy Curves
Confusion Matrix (Test Set)
Classification Report (Precision, Recall, F1-score)
7.  How to Run the Project
Step 1: Install Dependencies
pip install torch torchvision matplotlib seaborn scikit-learn
Step 2: Run the Project
python main.py
8.  Project Structure
Neural-Networks-Project/
│
├── main.py
├── README.md
├── data/ (auto-downloaded)
└── results/ (optional saved plots)
9.  Conclusion

This project demonstrates the effectiveness of a simple MLP neural network in solving image classification problems.
The experiments show that:

Activation functions significantly affect performance
Learning rate impacts stability and convergence
Proper regularization improves generalization

 Author

Neural Networks Course Project
Fashion-MNIST Classification using PyTorch
