# Deep Learning-Based Brand Classification & Visual Recommendation System

## 📌 Overview
This project implements a deep learning-based system for vehicle brand classification and visual similarity-based recommendations using the Stanford Cars Dataset.  
It combines VGG-19 for high-accuracy image classification with a Triplet Loss approach for building a visual recommendation system.

Developed as a Graduation Project for the Computer Engineering Department at Fatih Sultan Mehmet Vakıf University, this system aims to provide an accurate, automated, and scalable solution for analyzing large-scale vehicle image datasets.

## 🎯 Features
- Brand Classification: Classifies cars into their respective brands.
- Visual Recommendation: Suggests visually similar cars using embedding vectors.
- Custom Dataset Handling: Stanford Cars dataset reorganized into a brand-based classification format.
- Deep Learning Models:
  - Custom CNN for initial testing
  - VGG-19, ResNet-50, AlexNet for transfer learning
- Triplet Loss Integration for improved similarity search performance.
- Performance Evaluation: Accuracy, precision, recall, F1-score, and confusion matrix.
- GPU-Accelerated Training with PyTorch.

## 📂 Dataset
- Source: Stanford Cars Dataset
- Size: 16,185 images, 196 car models
- Custom Preprocessing:
  - Grouped images by brand instead of model
  - Normalization of pixel values
  - Data augmentation (rotation, horizontal flip, zoom, color changes)
- Split Ratios:
  - Training: 70%
  - Validation: 15%
  - Test: 15%

## 🛠 Tech Stack
- Programming Language: Python
- Deep Learning Frameworks: PyTorch, TensorFlow
- Image Processing: OpenCV
- Data Handling: NumPy, Pandas
- Visualization: Matplotlib
- Optimization: Adam Optimizer with learning rate scheduling

## 🧠 Model Architecture
### 1. VGG-19 (Pretrained)
- Used as the primary classification model.
- Modified fully connected layers for brand classification.
- Softmax activation for multi-class output.

### 2. Triplet Loss for Visual Recommendation
- Anchor: Reference image
- Positive: Same brand
- Negative: Different brand
- Objective: Minimize distance between Anchor & Positive, maximize distance between Anchor & Negative.

## 📊 Results
| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Custom CNN            | 81.45%   | 81.60%    | 81.45% | 81.50%   |
| VGG-19                | 94.02%   | 94.10%    | 94.02% | 94.05%   |
| VGG-19 + Triplet Loss | 96.15%   | 96.20%    | 96.15% | 96.17%   |

- VGG-19 achieved the best classification accuracy among tested models.
- Triplet Loss enabled a functional visual recommendation system.

## Screenshots

<img width="1470" height="853" alt="image" src="https://github.com/user-attachments/assets/eba9cb8f-bd5f-482f-a60c-32810a226340" /><br><br>


<img width="1414" height="742" alt="image" src="https://github.com/user-attachments/assets/973fb913-b1fe-43c9-b523-5484b6555ddf" /><br><br>


<img width="677" height="891" alt="image" src="https://github.com/user-attachments/assets/ab126ec2-7e3a-47da-aa0f-299e3383b524" /><br><br>

<img width="647" height="915" alt="image" src="https://github.com/user-attachments/assets/07de24be-1e64-4063-baa8-7105fe9f6168" /><br><br>


## 📌 Future Improvements

- Integrate transformer-based architectures for enhanced feature extraction.

- Improve handling of visually similar car models to reduce misclassification.


## 👨‍💻 Authors

Tarik Alrayan

Obada Masri

Ahmed Muaz Atik

Supervisor: Dr. Öğr. Üyesi Zeki KUŞ
