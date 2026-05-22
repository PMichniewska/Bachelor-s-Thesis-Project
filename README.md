# Machine and Deep Learning Approaches to Heart Rate Estimation from Speech with Interpretability and Fairness Analysis

**Author:** Patrycja Michniewska  
**Supervisor:** Dr. Wim Pouw  
**Institution:** Tilburg University, School of Humanities and Digital Sciences, Department of Cognitive Science and Artificial Intelligence  
**Submitted:** May 2026  

---

## Overview

This repository contains the full implementation for a thesis investigating whether heart rate (HR) in beats per minute (BPM) can be estimated from spontaneous and read speech recordings using machine learning. Two models were trained and evaluated under a strict speaker-independent protocol: **XGBoost** and a **Convolutional Neural Network Regressor (CNN-R)**. Secondary objectives included interpretability analysis using SHAP and fairness evaluation across demographic subgroups (gender, language, and BPM group).

The dataset used is the **Tamil and English Speech Database for Heartbeat Estimation (TESDHE)** by Milton and Monsely (2018), containing 10,040 speech recordings from 109 participants paired with manually annotated heart rate values.

Neither model surpassed the mean predictor baseline (test recording-level RMSE ≈ 16 BPM vs. baseline of 15.65 BPM). SHAP analysis indicated overreliance on higher-order distributional statistics, consistent with noise exploitation rather than genuine physiological signal. This repository serves as a reproducible diagnostic framework for future research in this area.

---

## Research Question

> To what extent can machine learning models, specifically XGBoost and a convolutional neural network regressor, reliably estimate heart rate from speech-derived features, and what do feature importance and subgroup performance evaluation reveal about their interpretability and fairness?

---

## Repository Structure

```
Thesis_local/
├── Implementation/
│   ├── Data Extraction Pipeline.ipynb     # Feature extraction from raw audio
│   ├── Coding Pipeline.ipynb              # Model training, evaluation, SHAP
│   ├── XGBoost_New_Normalisation.ipynb    # Post-hoc: XGBoost with CMVN normalisation
│   ├── XGBoost_Recording_Level.ipynb      # Post-hoc: XGBoost trained at recording level
│   └── Trial_feat_extraction.ipynb        # Exploratory feature extraction (before implementing in the pipeline)
│
├── data/
│   # Files not included
│
└── outputs/
    ├── exploration/                        # EDA plots and summary CSVs
    ├── baseline_results.csv               # Mean-predictor baseline metrics
    ├── CNN/                               # CNN-R outputs
    │   ├── cnn_best_params.json           # Best hyperparameters from random search
    │   ├── cnn_best_state.pt              # Saved model weights (PyTorch)
    │   ├── cnn_all_metrics.json           # Full train/val/test metrics
    │   ├── cnn_cv_fold_metrics_*.csv      # Per-fold CV metrics (frame and recording level)
    │   ├── cnn_cv_summary_*.csv           # CV mean ± SD summary tables
    │   ├── cnn_scatter_plots.png          # True vs. predicted BPM (val and test)
    │   ├── cnn_residual_plots.png         # Residual plots (val and test)
    │   ├── cnn_perfold_rmse.png           # Per-fold RMSE bar chart
    │   ├── cnn_bpm_distributions.png      # BPM distribution across train/val/test splits
    │   ├── cnn_rmse_by_bpm.png            # RMSE by BPM tertile (cross-validation)
    │   ├── cnn_*_heatmap_lang_gender.png  # RMSE heatmap by language × gender
    │   ├── cnn_shap_summary.png           # SHAP beeswarm plot (top 20 features)
    │   ├── cnn_shap_bar.png               # SHAP mean |SHAP| bar chart
    │   └── cnn_*_recording_predictions.csv# Per-recording predictions (val and test)
    ├── XGB/                               # XGBoost outputs
    │   ├── xgb_best.json                  # Saved best model
    │   ├── xgb_best_params.json           # Best hyperparameters
    │   ├── xgb_all_metrics.json           # Full train/val/test metrics
    │   ├── xgb_cv_*/  xgb_scatter_*/      # Same structure as CNN/ above
    │   ├── xgb_feature_importance.png     # Top 20 features by gain
    │   ├── xgb_shap_summary.png           # SHAP beeswarm plot
    │   ├── xgb_shap_bar.png               # SHAP mean |SHAP| bar chart
    │   └── xgb_speaker_5_*/               # Single-speaker post-hoc model outputs
    ├── new_norm/                          # XGBoost outputs with CMVN normalisation
    │   └── xgb_cmvn_*/                    # CMVN-specific metrics and model files
    └── per_rec/                           # XGBoost trained at recording level
        └── xgb_rec_*/                     # Recording-level model and metrics
```

---

## Data Access

> **The raw audio recordings and extracted features are not included in this repository.**

Access is subject to an End User License Agreement (EULA) signed with the dataset authors and the files exceed practical repository size limits. To obtain the data, contact the original authors:

> Milton, A., & Monsely, K. A. (2018). Tamil and English speech database for heartbeat estimation. *International Journal of Speech Technology, 21*(4), 967–973. https://doi.org/10.1007/s10772-018-9557-y

---

## Methods Summary

| Component | Details |
|---|---|
| Features | 20 MFCCs, F0 (pitch), local jitter, local shimmer, and per-feature entropy, skewness, kurtosis over 100-frame windows (92 features total) |
| Feature extraction | librosa 0.11 (MFCCs), openSMILE 2.6 with eGeMAPS v02 (prosodic) |
| Normalisation | Per-speaker MinMax scaling; post-hoc: CMVN per recording |
| Data split | Speaker-independent 70/15/15 (train/val/test), stratified by gender × language × BPM tertile |
| Cross-validation | 10-fold StratifiedGroupKFold (speaker as group variable) |
| Models | XGBoost (Chen & Guestrin, 2016); CNN-R following Ankışhan (2019) with dual pooling |
| Hyperparameter tuning | Coarse RandomizedSearchCV followed by fine GridSearchCV, scored on recording-level RMSE |
| Evaluation metrics | RMSE, MAE, R² at frame and recording level; subgroup breakdowns by gender, language, BPM tertile |
| Interpretability | SHAP TreeExplainer (XGBoost), GradientExplainer (CNN-R) |
| Fairness | Subgroup RMSE comparison; Bounded Group Loss considered but not applied due to insufficient baseline performance |

---

## Key Results

| Model | Test recording RMSE | Test R² | CV recording RMSE |
|---|---|---|---|
| Mean predictor (baseline) | 15.648 BPM | −0.008 | — |
| XGBoost | 15.99 BPM | −0.053 | 16.70 ± 1.55 BPM |
| CNN-R | 16.01 BPM | −0.055 | 17.01 ± 1.51 BPM |

Neither model surpassed the mean predictor. Both exhibited regression to the mean, with markedly higher RMSE for low- and high-BPM recordings. SHAP analysis revealed overreliance on distributional statistics (skewness, kurtosis, entropy), consistent with noise exploitation rather than physiological signal.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `Data Extraction Pipeline.ipynb` | Loads raw audio, extracts MFCCs and prosodic features, computes window-level statistics, applies normalisation, produces staged Parquet files |
| `Coding Pipeline.ipynb` | Loads normalised features, trains XGBoost and CNN-R with hyperparameter search, runs 10-fold CV, evaluates subgroup performance, produces SHAP plots |
| `XGBoost_New_Normalisation.ipynb` | Post-hoc experiment: replaces MinMax with CMVN normalisation and re-evaluates XGBoost |
| `XGBoost_Recording_Level.ipynb` | Post-hoc experiment: aggregates frame-level features to recording level (mean and SD) and re-evaluates XGBoost |

Run notebooks in the order listed above. `Coding Pipeline.ipynb` can be run independently if the staged Parquet files are present.

---

## Requirements

Python 3.12. Install dependencies with:

```bash
pip install -r requirements.txt
```


> Experiments were run on an Apple MacBook Pro with an M1 chip. GPU acceleration was not used. Training the CNN-R with hyperparameter search may take several hours on CPU.

---

## Reproducing the Results

1. Obtain the TESDHE audio data (see Data Access above) and place it under `data/TESDHE/`.
2. Run `Data Extraction Pipeline.ipynb` to reproduce all staged Parquet files. Skip this step if the Parquet files are already present.
3. Run `Coding Pipeline.ipynb` to train both models, evaluate performance, and produce SHAP outputs.
4. Optionally run `XGBoost_New_Normalisation.ipynb` and `XGBoost_Recording_Level.ipynb` for post-hoc experiments.

All random seeds are fixed (seed = 42) throughout to ensure reproducibility.

---

## License

This repository is shared for academic and research purposes. The TESDHE dataset is subject to its own EULA and is not covered by this license.

---

## Acknowledgements

This project was completed under the supervision of Dr. Wim Pouw at Tilburg University. Computational resources were provided by the researcher's own hardware. Claude (Sonnet 4.6, Anthropic) was used to assist in debugging and reviewing individual code segments.
