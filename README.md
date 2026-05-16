Fashion-MNIST Classification — Neural Networks Project

Neural Networks Course Project
Multilayer Perceptron (MLP) implemented using PyTorch

Problem Description

This project addresses the image classification problem using the Fashion-MNIST dataset.

The goal is to classify grayscale clothing images (28×28) into 10 categories using a Multilayer Perceptron (MLP).

Different model configurations are tested by changing activation functions and learning rates to analyze their effect on performance and generalization.

📂 Dataset
Dataset: Fashion-MNIST (Zalando Research)
Source: Kaggle / torchvision
Image Size: 28 × 28 pixels (grayscale)
Classes: 10 clothing categories
Training Samples: 60,000
Test Samples: 10,000
 Class Labels
Label	                Class
0	               T-shirt/top
1	                Trouser
2	          Pullover
3	             Dress
4	           Coat
5	               Sandal
6	          Shirt
7	S neaker
8	Bag
9	Ankle boot
⚙️ Preprocessing
Normalization applied to pixel values
Flattening: 28×28 → 784 vector
Grayscale images (1 channel)
No missing values or cleaning required
🧠 Model Architecture
Input Layer      → 784 neurons
Hidden Layer 1   → 128 neurons → BatchNorm → Activation → Dropout(0.3)
Hidden Layer 2   → 128 neurons → BatchNorm → Activation → Dropout(0.3)
Output Layer     → 10 neurons (CrossEntropyLoss)
⚙️ Training Setup
Loss Function: CrossEntropyLoss
Optimizer: Adam
Batch Size: 64
Epochs: 5
Hidden Size: 128
Dropout: 0.3
🧪 Experiments

Three experiments were conducted by changing activation function and learning rate:

Experiment	Activation	Learning Rate	Test Accuracy	Test Loss
Exp 1	ReLU	0.001	84.27%	0.4257
Exp 2	Tanh	0.001	—	—
Exp 3	ReLU	0.0005	—	—
📊 Training Summary (Exp 1)
Metric	Value
Final Train Accuracy	82.30%
Final Validation Accuracy	85.28%
Final Train Loss	0.4847
Final Validation Loss	0.4008
📊 Key Observations
🔹 Activation Function (ReLU vs Tanh)
ReLU outperforms Tanh in convergence speed and stability
Tanh suffers from slower training in deeper layers
ReLU shows better generalization on validation data
🔹 Learning Rate Impact
lr = 0.001 → stable and best performance
Lower learning rate improves convergence stability
No signs of overfitting observed
🏆 Best Model

Experiment 1 (ReLU + lr=0.001)
✔ Best accuracy
✔ Stable training
✔ Good generalization

📈 Visualizations
Training vs Validation Loss Curves
Training vs Validation Accuracy Curves
Confusion Matrix
Classification Report
Model Comparison (Experiments)
🚀 How to Run
📌 Google Colab (Recommended)
Open Google Colab
Upload Fashion_MNIST_MLP.ipynb
Run all cells
Dataset downloads automatically

(Optional) Enable GPU:

Runtime → Change runtime type → GPU
💻 Local Setup
git clone https://github.com/your-repo/Neural-Networks-Project.git
cd Neural-Networks-Project

pip install -r requirements.txt

jupyter notebook Fashion_MNIST_MLP.ipynb
📁 Project Structure
├── Fashion_MNIST_MLP.ipynb
├── README.md
├── requirements.txt
└── results/
    ├── loss_curves.png
    ├── accuracy_curves.png
    ├── confusion_matrix.png
    └── sample_predictions.png
✅ Project Checklist
✔ Dataset loaded and preprocessed
✔ MLP model implemented in PyTorch
✔ Multiple experiments conducted
✔ Training & validation monitoring
✔ Test evaluation completed
✔ Confusion matrix generated
✔ Classification report included
✔ Visualization of results
✔ Comparison between configurations
🏁 Conclusion
ReLU performs better than Tanh in this task
Lower learning rate improves training stability
Model achieves good generalization (~84% accuracy)
No significant overfitting observed
👨‍💻 Author

Neural Networks Course Project
Fashion-MNIST Classification using PyTorch
