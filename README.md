# Dog Breed Classification using CNN + Transformer
This project classifies dog breeds from images using a hybrid deep learning approach that combines a CNN (Xception) feature extractor and a Transformer for sequence modeling. The model is trained and fine-tuned on the Stanford Dogs dataset (120 classes), achieving robust results for multi-class classification.

## 🚀 Features
CNN Backbone: Xception pre-trained on ImageNet for feature extraction.
Transformer Layers: Captures long-range dependencies and refines feature embeddings.
Early Stopping: Prevents overfitting by monitoring validation loss.
Input Resolution: 224x224 RGB images.
Number of Classes: 120 Dog Breeds.
Top-5 Predictions available for inference.

## 📋 Architecture
Feature Extraction:
Xception Model (without final classification layers).

Feature Sequences:
Output from CNN reshaped as a sequence.

Transformer Encoder:
Processes the feature sequence.

Classification Head:
Final linear layers for predicting 120 dog breeds.

## 🐕 Dataset
Source: Stanford Dogs Dataset (https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)

Classes: 120 Dog Breeds

Training/Validation/Test Split:
Total Images: 20,580 (combined from both the train and test folders of the dataset, shuffled, and split into 10 equal parts)

Training Set: 7 parts (7 × 2,058) = 14,406 images

Validation Set: 1 part (2,058 images)

Test Set: 2 parts (2 × 2,058) = 4,116 images

Input Image Size: 224×224 pixels

Labeling:
labels_r dictionary maps integer labels (0–119) to breed names.

## ⚡️ Training Details
Framework: PyTorch

Model Backbone: Xception

Transformer Layers: Self-attention + Feedforward

Optimizer: AdamW

Learning Rate: Tuned using RandomizedSearchCV (initial LR ~1e-4).

Early Stopping: Monitors val_loss.

Training Epochs: Approximately 20, with early stopping to halt training when no further improvement is observed

Final Model: xception_transformer_best.pt

## ✅ Results

Achieved %93.46 accuracy

Our model ranks 5th among other works evaluated on the same dataset and task

![top5new](https://github.com/user-attachments/assets/8d11ba50-f6b8-499b-8918-b082fd289b1c)
