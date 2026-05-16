 Neural Networks Project – MLP (Fashion-MNIST)
 Problem Description

This project implements a Multilayer Perceptron (MLP) model to classify images from the Fashion-MNIST dataset into 10 categories of clothing items such as T-shirts, trousers, shoes, etc.

The main objective of this project is to study neural network behavior and analyze how different hyperparameters affect model performance in terms of accuracy and loss.

 Dataset
Name: Fashion-MNIST
Source: https://github.com/zalandoresearch/fashion-mnist
Type: Image Classification Dataset
Image Size: 28 × 28 grayscale
Number of Classes: 10

The dataset is split into training, validation, and testing sets.
A validation split is created from the training data (80/20 split).

 Model Architecture

A Multilayer Perceptron (MLP) is used with the following structure:

Input Layer: 784 neurons (flattened 28×28 images)
Hidden Layer 1: hidden_size neurons (128 or 256)
Hidden Layer 2: hidden_size / 2 neurons (64 or 128)
Output Layer: 10 neurons (one per class)
 Activation Functions
ReLU (Experiment 1 & 3)
Tanh (Experiment 2)
 Loss Function
CrossEntropyLoss
 Optimizer
Adam optimizer
 Data Preprocessing

The following preprocessing steps were applied:

Normalization using mean = 0.5 and std = 0.5
Data Augmentation:
Random rotation
Random horizontal flip

These techniques help improve model generalization and reduce overfitting.

 Experiments

Three experiments were conducted to analyze the effect of different hyperparameters on model performance.

Experiment	Activation	Hidden Size	Learning Rate	Description
Exp 1	ReLU	128	0.001	Baseline model
Exp 2	Tanh	128	0.001	Effect of activation change
Exp 3	ReLU	128	0.0005	Effect of learning rate change
 Experiment Strategy

Each experiment changes only one parameter at a time to ensure a fair and scientific comparison.

📈 Results & Observations
Training loss decreases steadily over epochs
Validation accuracy improves consistently
Model performance is sensitive to activation function and learning rate
ReLU generally converges faster than Tanh
 Final Results
Test accuracy varies per experiment depending on hyperparameters
Overall performance range is approximately consistent across runs
 Comparison Table
Experiment	               Accuracy	    Loss
Exp 1 (ReLU, 128, lr=0.001)	~88%	    Higher
Exp 2 (Tanh, 128, lr=0.001)	~86–87%	  Medium
Exp 3 (ReLU, 128, lr=0.0005)	~89%	   Lower
 Visualizations

The following plots are included in the project:

Training vs Validation Loss Curves
Training vs Validation Accuracy Curves

These visualizations help analyze:

Learning behavior over epochs
Overfitting / underfitting patterns
Comparison between experiments
 Techniques Used
 Dropout

Used to reduce overfitting by randomly disabling neurons during training.

 Batch Normalization

Applied after fully connected layers to stabilize training and improve convergence speed.

 Data Augmentation

Used to increase dataset diversity and improve generalization.

 Hyperparameter Tuning

The following parameters were tested:

Activation functions
Learning rate
Hidden layer size
 How to Run the Project
pip install torch torchvision matplotlib seaborn scikit-learn
python project.py
 Project Structure
Neural-Networks-Project/
project.py
requirements.txt
 README.md
data/   (auto downloaded)

 Dataset Source

https://github.com/zalandoresearch/fashion-mnist
