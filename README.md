Dog Breed Classification using CNN + Transformer
This project classifies dog breeds from images using a hybrid deep learning approach that combines a CNN (Xception) feature extractor and a Transformer for sequence modeling. The model is trained and fine-tuned on the Stanford Dogs dataset (120 classes), achieving robust results for multi-class classification.

## 🚀 Features
CNN Backbone: Xception pre-trained on ImageNet for feature extraction.

Transformer Layers: Captures long-range dependencies and refines feature embeddings.

Early Stopping: Prevents overfitting by monitoring validation loss.

Input Resolution: 122x122 RGB images.

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
Source: Stanford Dogs Dataset

Classes: 120 Dog Breeds

Training/Validation Split:

Training: ~12k images

Validation: ~8k images

Input Image Size: 122×122

Labeling:

labels_r dictionary maps integer labels (0–119) to breed names.

## ⚡️ Training Details
Framework: PyTorch

Model Backbone: Xception

Transformer Layers: Self-attention + Feedforward

Optimizer: AdamW

Learning Rate: Tuned using RandomizedSearchCV (initial LR ~1e-4).

Early Stopping: Monitors val_loss.

Training Epochs: ~25–30

Final Model: xception_transformer_best.pt
