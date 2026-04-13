# Heart Rate Estimation from Speech using Machine Learning

## Overview
This project investigates whether heart rate (in beats per minute, BPM) can be estimated from speech recordings using machine learning and deep learning techniques. It combines signal processing, regression modelling, interpretability, and fairness analysis.

The work is part of an ongoing Bachelor’s thesis titled:

**"Machine and Deep Learning Approaches to Heart Rate Estimation from Speech with Interpretability and Fairness Analysis"**

## Objectives
- Estimate heart rate from speech recordings using acoustic features
- Compare performance of machine learning and deep learning models
- Analyse feature importance for interpretability
- Evaluate and mitigate potential algorithmic bias across demographic groups

## Dataset
The project uses the *Tamil and English Speech Database for Heartbeat Estimation*, which contains:
- ~10,000 speech recordings
- 109 participants (male and female, ages 11–50)
- Annotated heart rate values (BPM)
- Multiple conditions (baseline, post-activity, emotional)

## Methodology

### 1. Preprocessing
- Audio cleaning (duplicate removal)
- Resampling and normalization
- Segmentation into short frames (20 ms with 50% overlap)

### 2. Feature Extraction
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Fundamental frequency (F0 / pitch)
- Entropy, skewness, kurtosis
- Jitter and shimmer

### 3. Models
The task is formulated as a regression problem. Models include:
- XGBoost (primary baseline)
- Convolutional Neural Network for Regression (CNN-R)

### 4. Evaluation
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score
- Speaker-independent cross-validation

### 5. Interpretability
- SHAP (SHapley Additive Explanations) for feature importance Analysis of how acoustic features contribute to predictions

### 6. Fairness Analysis
- Performance comparison across subgroups:
  - Gender
  - Language (Tamil vs English)
  - Age groups
- Bias mitigation using Bounded Group Loss (FairLearn)

## Status
**Work in Progress**

- ✅ Literature review
- ✅ Methodology design
- ✅ Data preprocessing pipeline
- ✅ Feature extraction
- 🔄 Model development and evaluation
- ⏳ Interpretability and fairness analysis

## Potential Applications
- Contactless health monitoring
- Early detection of cardiovascular issues
- Deepfake audio detection via physiological signals

## Author
Patrycja Michniewska  
Tilburg University – Cognitive Science and Artificial Intelligence

