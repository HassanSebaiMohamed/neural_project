Neural Networks Project – MLP (Fashion-MNIST)
Problem Description

This project implements a Multilayer Perceptron (MLP) model to classify images from the Fashion-MNIST dataset into 10 categories of clothing items such as T-shirts, trousers, shoes, etc.

The goal is to study neural network performance and analyze how different hyperparameters affect accuracy and loss.

Dataset
Name: Fashion-MNIST
Link: https://github.com/zalandoresearch/fashion-mnist
Type: Image Classification Dataset
Image Size: 28 × 28 grayscale
Number of Classes: 10
Model Architecture

A Multilayer Perceptron (MLP) is used with the following structure:

Input Layer: 784 neurons (flattened 28×28 images)
Hidden Layer 1: 128 / 256 neurons
Hidden Layer 2: 64 / 128 neurons
Output Layer: 10 neurons (classification)
Activation Functions:
ReLU (Experiment 1)
Tanh (Experiment 2)
Loss Function:
CrossEntropyLoss
Optimizer:
Adam
Data Preprocessing
Images are normalized using mean = 0.5 and std = 0.5
Data augmentation applied:
Random rotation
Random horizontal flip
Train/Test split is provided by dataset automatically
Experiments

Two experiments were conducted by changing activation function, hidden layer size, and learning rate.

Experiment	Activation	Hidden Size	Learning Rate	Description
Exp 1	ReLU	128	0.001	Baseline model
Exp 2	Tanh	256	0.0005	Improved model capacity
Results
Training loss decreases over epochs
Accuracy increases during training
Experiment 2 shows better performance due to higher model capacity and smoother activation
Final Results (Example Output):
Test Accuracy: approximately 87% - 90%
Final Loss: depends on training run
Visualizations

The following plots are included:

Training Loss Curve
Accuracy Curve

These graphs show:

How the model learns over time
Comparison between experiments
Techniques Used
Dropout

Used to reduce overfitting by randomly disabling neurons during training.

Data Augmentation

Used to increase dataset diversity and improve generalization.

Hyperparameter Tuning

Different values of:

Activation functions
Hidden layer sizes
Learning rate

were tested to analyze their effect on performance.

How to Run the Project

pip install torch torchvision matplotlib
python project.py

Project Structure

Neural-Networks-Project/
│
├── project.py
├── requirements.txt
├── README.md
└── data/ (auto downloaded)

Conclusion

This project demonstrates the implementation of a Multilayer Perceptron for image classification. It shows how architectural choices and hyperparameters affect model performance. Experimentation confirms that increasing model capacity and tuning learning rate improves results.

Dataset Source

https://github.com/zalandoresearch/fashion-mnist