# Credit Card Fraud Detection using PyTorch

## Overview
This project uses PyTorch to build and train a deep learning model for detecting credit card fraud. The dataset is highly imbalanced, so SMOTE (Synthetic Minority Over-sampling Technique) is employed for effective data balancing. The model includes an autoencoder for dimensionality reduction and achieves high accuracy, precision, and recall.

## Features
- Balanced dataset using SMOTE
- Autoencoder for feature encoding
- PyTorch-based neural network with dropout for improved generalization
- Early stopping to prevent overfitting
- Achieves 100% precision, recall, and F1-score on the test set
- Visualized metrics for clear model evaluation

## Dataset
The dataset includes:
- **Non-fraud data count:** 1,836,743
- **Fraud data count:** 3,651
- After applying SMOTE:
  - Class 0 (Non-fraud): 1,836,743
  - Class 1 (Fraud): 1,836,743

## Model Architecture
### Autoencoder (Feature Encoding)
- **Input Layer:** 22 features
- **Encoder:** 22 → 64 → 7
- **Decoder:** 7 → 64 → 22
- **Activation Function:** ReLU and Sigmoid

### Main Model (Fraud Detection)
- Input layer with 7 encoded features
- Hidden layers: 64 → 32 → 16
- Dropout for improved generalization
- Output layer with sigmoid activation for binary classification

## Requirements
- Python 3.x
- PyTorch
- TorchSummary
- Imbalanced-learn (for SMOTE)

Install dependencies with:
```bash
pip install torch torchsummary imbalanced-learn
```

## Training Process
1. **Data Preprocessing**
   - SMOTE applied to balance the dataset
   - Autoencoder for feature encoding
   - Data scaled and split into training, validation, and test sets
2. **Model Training**
   - Loss function: Binary Cross-Entropy Loss
   - Optimizer: Adam
   - Learning rate scheduler included
3. **Evaluation**
   - Precision, Recall, and F1-score calculated
   - AUC-ROC and AUC-PR metrics for robust evaluation

## Results
- **Train Accuracy:** 99.99%
- **Validation Accuracy:** 100%
- **Test Accuracy:** 100%
- **Precision, Recall, F1-Score:** 1.0000
- **AUC-ROC:** 1.0000

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```
2. Navigate to the project directory:
```bash
cd credit-card-fraud-detection
```
3. Run the training script:
```bash
python train.py
```

## Model Checkpoint
The trained model is saved as `model_checkpoint.pth`. To load the model:
```python
import torch
model = torch.load('model_checkpoint.pth')
model.eval()
```

## Evaluation
To evaluate the model on test data:
```bash
python evaluate.py
```

## Contributing
Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions or inquiries, please contact [your_email@example.com].

