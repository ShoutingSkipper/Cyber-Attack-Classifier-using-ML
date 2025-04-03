# Cyber Attack Classification Using Machine Learning

![Cyber Security Banner](https://img.shields.io/badge/Project-Cyber%20Security-blue) ![Python](https://img.shields.io/badge/Language-Python-3.11-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

Welcome to the **Cyber Attack Classification** project! This repository contains a machine learning solution to detect cyber-attacks using the NSL-KDD dataset. Built with Python, it leverages advanced techniques like PCA for dimensionality reduction and XGBoost for ensemble learning to classify network traffic as "normal" or "attack" with high accuracy.

---

## Executive Summary

This project develops a robust **Intrusion Detection System (IDS)** by training machine learning models to identify cyber-attacks from network traffic data. Using the NSL-KDD dataset, we preprocess 41 features, apply feature reduction, and evaluate multiple classifiers—Decision Tree, Random Forest, SVM, and XGBoost—achieving an impressive **99.9% accuracy** with XGBoost. The purpose is to automate attack detection, making it a valuable tool for network security, anomaly detection, and cybersecurity research.

### Highlights
- **High Accuracy**: Achieved 99.9% accuracy with XGBoost on original data.
- **Dimensionality Reduction**: PCA reduces features to 10 components, maintaining strong performance.
- **Ensemble Power**: XGBoost outperforms traditional models, enhancing detection reliability.
- **Comprehensive Evaluation**: Metrics include Accuracy, Precision, Recall, and F1-Score, with visualizations like confusion matrices and PCA scatter plots.

---

## Project Overview

### Objective
Train and evaluate machine learning models to classify network traffic as "normal" (0) or "attack" (1) using the NSL-KDD dataset, fulfilling the requirements of an academic assignment while demonstrating practical cybersecurity applications.

### Dataset
- **Source**: NSL-KDD (`KDDTrain+.csv`), a benchmark dataset for intrusion detection.
- **Features**: 41 network traffic attributes (e.g., `src_bytes`, `protocol_type`, `serror_rate`) + 1 label (`class`) + 1 extra column.
- **Size**: ~125,973 records, with a mix of normal and attack instances.

### Use Case
The model serves as an **Intrusion Detection System (IDS)** with real-world applications:
- **Network Security**: Detects attacks like DoS, probing, and unauthorized access in real-time.
- **Anomaly Detection**: Identifies unusual traffic patterns, including zero-day attacks.
- **Forensics**: Analyzes historical logs for threat intelligence.
- **Research**: Benchmarks ML techniques for cybersecurity advancements.
- **Automation**: Triggers defensive actions (e.g., blocking IPs) in SOAR platforms.

---

## Methodology

### 1. Dataset Preparation
- **Loading**: Read `KDDTrain+.csv` using Pandas, assigning 43 column names (41 features + `class` + `extra`).
- **Label Conversion**: Transformed `class` into binary labels: `normal` → 0, all attacks → 1.
- **Encoding**: Applied Label Encoding to categorical features (`protocol_type`, `service`, `flag`).
- **Normalization**: Scaled numerical features using MinMaxScaler for consistency.

### 2. Exploratory Data Analysis (EDA)
- **Stats**: Checked dataset shape, missing values, and class distribution.
- **Visualization**: Plotted class distribution with Seaborn to highlight imbalance (normal vs. attack).
- **Correlation**: Identified top 5 features correlated with `class` (e.g., `serror_rate`, `count`).

### 3. Classification Models
- **Models Trained**:
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- **Process**:
  - Split data: 80% train, 20% test.
  - Trained each model on original features.
  - Evaluated using Accuracy, Precision, Recall, F1-Score, and confusion matrices.
- **Output**: Random Forest and SVM showed strong results, setting a baseline.

### 4. Dimensionality Reduction
- **Technique**: Applied Principal Component Analysis (PCA) to reduce features to 10 components.
- **Retraining**: Re-trained Random Forest on PCA-reduced data.
- **Comparison**: Compared performance (e.g., F1-Score) before and after PCA, showing minimal accuracy loss with faster computation.

### 5. Ensemble Learning
- **Technique**: Implemented XGBoost, an advanced ensemble method.
- **Training**: Trained XGBoost on both original and PCA-reduced data.
- **Results**: Achieved highest performance (Accuracy: 99.9%, F1-Score: 0.999) on original data, with PCA version still```python
still slightly lower but robust (F1-Score: 0.998).

### 6. Visualizations
- **Confusion Matrices**: Plotted for each model to visualize true positives/negatives.
- **PCA Scatter Plot**: 2D projection of data post-PCA, showing class separation.
- **F1-Score Bar Plot**: Compared all models’ performance.

---

## Getting Started

### Prerequisites
- Python 3.11+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `seaborn`, `matplotlib`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cyber-attack-classification.git
   cd cyber-attack-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Upload `KDDTrain+.csv` to your working directory.

### Running the Code
- Open `cyber_attack_classification.ipynb` in Jupyter Notebook or Google Colab.
- Execute all cells to preprocess data, train models, and generate results.

---

## Results
| Model              | Accuracy  | F1-Score  |
|--------------------|-----------|-----------|
| Decision Tree      | 0.995     | 0.994     |
| Random Forest      | 0.998     | 0.998     |
| SVM                | 0.997     | 0.996     |
| Random Forest (PCA)| 0.996     | 0.995     |
| XGBoost            | 0.999     | 0.999     |

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- NSL-KDD dataset creators for providing a robust benchmark.
- xAI for inspiring AI-driven solutions like this one.

---
