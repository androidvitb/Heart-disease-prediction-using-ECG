# Heart Disease Prediction Using ECG and Vision Transformer (ViT)

This project combines machine learning and computer vision techniques to analyze electrocardiogram (ECG) data and classify heart conditions into four categories:

1. **Myocardial Infarction**
2. **Abnormal Heartbeat**
3. **History of Myocardial Infarction**
4. **Normal Heartbeat**

The project uses the Vision Transformer (ViT) model fine-tuned on ECG-XRAY images and is hosted on the Hugging Face Model Hub.

---

### **Repository Under**: AcWoC'25  
### **Club**: Android Club, VIT Bhopal University  

---

## Table of Contents
1. [Dataset](#dataset)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Training the Model](#training-the-model)
6. [Evaluating the Model](#evaluating-the-model)
7. [Classifying User-Provided Images](#classifying-user-provided-images)
8. [Model Deployment](#model-deployment)
9. [Results](#results)
10. [References](#references)
11. [Acknowledgments](#acknowledgments)

---

## Dataset

The dataset used for training and evaluation is located in the `Dataset` directory and can also be accessed from the Hugging Face Dataset Hub:  
[ECG-XRAY Dataset](https://huggingface.co/datasets/sagar27kumar/ECG-XRAY-dataset)

It consists of ECG recordings labeled according to the aforementioned categories.

---

## Features
- **Vision Transformer (ViT)**: Fine-tuned for medical image classification.
- **ECG Analysis**: Classifies images into four heart condition categories.
- **Evaluation Metrics**: Includes accuracy, precision, recall, and confusion matrices.
- **User Input Classification**: Supports real-time ECG image classification.
- **Model Hosting**: Published on Hugging Face Hub for easy reuse.

---

## Requirements

Ensure you have the necessary Python packages installed before running the project. Dependencies include:
- `transformers`
- `datasets`
- `torch`
- `Pillow`
- `scikit-learn`

Install them using:
```bash
pip install -r requirements.txt
