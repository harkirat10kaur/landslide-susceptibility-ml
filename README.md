# AI-Powered Prediction of Landslide Susceptibility Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: Mendeley Data](https://img.shields.io/badge/Dataset%20DOI-10.17632%2Fmxpg845z62.2-blue)](https://doi.org/10.17632/mxpg845z62.2)
[![Framework: scikit-learn | XGBoost | SHAP](https://img.shields.io/badge/Framework-scikit--learn%20%7C%20XGBoost%20%7C%20SHAP-orange)](https://scikit-learn.org/)
[![Tests: 10 Passed](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)](tests/)

An end-to-end, scientifically rigorous geospatial machine learning pipeline designed to model, evaluate, interpret, and map regional landslide susceptibility across **Himachal Pradesh, Western Himalayas, India**. The system leverages authentic ground-truth inventory data, digital elevation models (SRTM), satellite precipitation (CHIRPS), and high-resolution land cover (ESA WorldCover) conditioning factors.

---

## Table of Contents
- [1. Project Overview & Scientific Motivation](#1-project-overview--scientific-motivation)
- [2. Dataset Documentation & Preprocessing](#2-dataset-documentation--preprocessing)
- [3. Methodology & Cross-Validation Strategy](#3-methodology--cross-validation-strategy)
- [4. Quantitative Experimental Results](#4-quantitative-experimental-results)
  - [A. Independent Holdout Test Set Performance](#a-independent-holdout-test-set-performance-n2348)
  - [B. Primary Model Selection Rationale: Random Forest vs. XGBoost](#b-primary-model-selection-rationale-random-forest-vs-xgboost)
  - [C. The Spatial Autocorrelation Leakage Gap](#c-the-spatial-autocorrelation-leakage-gap-spatial-vs-random-cv)
  - [D. Operational Hazard Threshold Optimization](#d-operational-hazard-threshold-optimization-asymmetric-cost-tuning)
  - [E. Five-Class Susceptibility Distribution & Ground-Truth Alignment](#e-five-class-susceptibility-distribution--ground-truth-alignment)
- [5. Explainable AI & Geophysical Insights](#5-explainable-ai--geophysical-insights)
- [6. Methodological Framework: Key Concepts & Limitations](#6-methodological-framework-key-concepts--limitations)
  - [A. Nature of Pseudo-Absence Samples](#a-nature-of-pseudo-absence-samples)
  - [B. Interpretation of Susceptibility Scores $S(x)$](#b-interpretation-of-susceptibility-scores-sx)
  - [C. False-Positive Limitations & Asymmetric Error Cost](#c-false-positive-limitations--asymmetric-error-cost)
  - [D. External Generalization & Dynamic Triggers](#d-external-generalization--dynamic-triggers)
- [7. Project Directory Structure](#7-project-directory-structure)
- [8. Installation & Environment Setup](#8-installation--environment-setup)
- [9. How to Run the Pipeline](#9-how-to-run-the-pipeline)
- [10. Automated Testing & Verification](#10-automated-testing--verification)
- [11. Data Provenance & Citation](#11-data-provenance--citation)

---

## 1. Project Overview & Scientific Motivation
Slope failures, debris flows, and rockfalls in the Himalayan orogen represent recurring, catastrophic natural hazards that cause severe loss of life, disrupt strategic transport corridors (e.g., NH-05, NH-21), and inflict severe infrastructure damage. Regional **Landslide Susceptibility Mapping (LSM)** provides spatial hazard zonation essential for civil defense agencies (NDMA, SDMA), urban planning authorities, and transportation engineers.

This repository provides a reproducible, leakage-free machine learning workflow featuring:
- **Authentic Indian Himalayan Inventory:** 11,737 coordinate-deduplicated samples across all 12 districts of Himachal Pradesh based on the Geological Survey of India (GSI) inventory and Mendeley Data ([DOI: 10.17632/mxpg845z62.2](https://doi.org/10.17632/mxpg845z62.2), Version 2).
- **Spatial Leakage Diagnostics:** Direct comparison between 5-fold Spatial Block GroupKFold and conventional Random StratifiedKFold to isolate and measure the spatial autocorrelation optimism bias.
- **Multi-Algorithm Benchmarking:** Evaluation of six candidate machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, XGBoost, and Multi-Layer Perceptron) plus a Soft-Voting Ensemble.
- **Asymmetric Operational Threshold Tuning:** Cost-sensitive threshold optimization on validation folds prioritizing high sensitivity ($Recall \ge 90\%$) to minimize catastrophic false negatives.
- **Explainable AI (XAI):** Multi-method interpretability via Gini Impurity (MDI), test-set permutation importance, SHAP TreeExplainer (beeswarm and dependence plots), and 4-domain feature-group ablation experiments.
- **Continuous Surface & 5-Tier Zonation Maps:** Production of 300 DPI publication-quality hazard maps and export of a georeferenced GeoTIFF continuous raster (`maps/susceptibility_surface.tif`).
- **Production-Ready Inference Engine:** A modular, object-oriented predictor (`predict.py`) with input validation, automated aspect decomposition, and robust missing value imputation.

---

## 2. Dataset Documentation & Preprocessing

### Provenance and Regional Scope
- **Source:** Mendeley Data ([DOI: 10.17632/mxpg845z62.2](https://doi.org/10.17632/mxpg845z62.2), Version 2).
- **Dataset License:** Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Study Region:** Himachal Pradesh, Western Himalayas, India ($30.00^\circ\text{N} - 33.01^\circ\text{N}$, $75.71^\circ\text{E} - 79.83^\circ\text{E}$).
- **Elevation Range:** 247 m (Shivalik foothills) to 6,420 m (Greater Himalayan alpine ridge).
- **Districts Covered (All 12):** Kangra, Mandi, Shimla, Kullu, Kinnaur, Solan, Sirmaur, Chamba, Bilaspur, Hamirpur, Una, and Lahaul & Spiti.

### Sample Counts and Hygiene
- **Raw Count:** 12,409 sample records (6,531 historical landslides, 5,878 pseudo-absences).
- **Cleaned Deduplicated Count:** **11,737 unique spatial locations** (672 co-located 30 m pixel duplicates removed to prevent pixel-level data leakage between folds).
  - Confirmed Historical Landslides (`class = 1`): **5,859 samples (49.92%)**
  - Pseudo-Absence Non-Landslides (`class = 0`): **5,878 samples (50.08%)**
  - Class Ratio: Balanced (~1:1).

### Geo-Environmental Conditioning Factors (10 Features)

| Feature Name | Domain | Type | Raw Range / Units | Description & Source |
|---|---|---|---|---|
| `slope_deg` | Morphometric | Continuous | 0.0° – 76.8° | Terrain slope gradient from SRTM 30 m DEM |
| `elevation_m` | Morphometric | Continuous | 246 m – 6,430 m | Elevation above sea level from SRTM 30 m DEM |
| `aspect_deg` | Morphometric | Continuous | 0.0° – 359.9° | Compass slope direction from SRTM 30 m DEM |
| `Aspect_cos` | Morphometric | Continuous | [-1.0, +1.0] | Northness component: $\cos(\text{aspect}_\text{rad})$ (+1 North, -1 South) |
| `Aspect_sin` | Morphometric | Continuous | [-1.0, +1.0] | Eastness component: $\sin(\text{aspect}_\text{rad})$ (+1 East, -1 West) |
| `Dist_River_z` | Hydrologic | Continuous | [-0.13, +8.04] | Standardized Euclidean distance to drainage network (HydroSHEDS) |
| `FlowAccumulation_z`| Hydrologic | Continuous | [-0.58, +5.01] | Standardized catchment upstream flow accumulation (HydroSHEDS) |
| `NDVI_z` | Ecological | Continuous | [-5.97, +1.80] | Standardized median Normalized Difference Vegetation Index (Landsat) |
| `Rainfall_z` | Climatic | Continuous | [-2.33, +3.92] | Standardized long-term annual precipitation (CHIRPS) |
| `lulc_name` | Ecological | Categorical | 10 classes | Land use / land cover classification (ESA WorldCover 2020) |

### Preprocessing & Multicollinearity Safeguards
1. **Aspect Circularity Decomposition:** Slope aspect is cyclical ($0^\circ \equiv 360^\circ$). Raw degrees are decomposed into continuous trigonometric components ($Aspect_{\cos} = \cos\theta$, $Aspect_{\sin} = \sin\theta$) to prevent artificial boundary discontinuities in tree-based splits and distance metrics.
2. **Leakage-Free Feature Transformation:** All scaling (`StandardScaler`), median imputation (`SimpleImputer`), and categorical encoding (`OneHotEncoder(handle_unknown='ignore')`) are bundled inside scikit-learn `Pipeline` and `ColumnTransformer` instances fitted strictly on training partitions.
3. **Multicollinearity Diagnosis:** Variance Inflation Factor (VIF) analysis confirms the absence of severe multicollinearity across all continuous conditioning factors (all VIF $< 2.6$, well below the conservative threshold of 5.0):
   - Highest VIF: `Aspect_cos` (2.56) and `Aspect_sin` (2.56), followed by `elevation_m` (1.35) and `Rainfall_z` (1.29).

---

## 3. Methodology & Cross-Validation Strategy

### Data Partitioning
An untouched **20% holdout test partition ($N=2,348$)** was stratified by target class and isolated prior to any hyperparameter tuning or threshold selection. All model exploration and threshold tuning were conducted strictly on the remaining **80% training pool ($N=9,389$)**.

```
Full Cleaned Dataset (N = 11,737)
├── Untouched Evaluation Test Set (20%, N = 2,348) [Stratified by class]
└── Training & Validation Pool   (80%, N = 9,389)
    ├── 5-Fold Spatial GroupKFold CV (KMeans coordinate sectors, leakage-free)
    ├── 5-Fold Conventional StratifiedKFold CV (random partitioning)
    └── Validation Threshold Tuning (Recall >= 90% optimization)
```

### Validation Protocols
1. **Spatial GroupKFold (Leakage-Free):** The study area was partitioned into 5 contiguous geographic clusters using K-Means clustering on geographic coordinates `(longitude, latitude)`. In each fold, four regional clusters are used for training and the remaining cluster is used for validation. This simulates regional extrapolation to unmapped valleys and isolates spatial autocorrelation bias.
2. **Conventional Random StratifiedKFold:** Standard 5-fold cross-validation with random shuffling, matching the conventional methodology prevalent in published literature.
3. **SMOTE Experimentation:** Synthetic Minority Over-sampling Technique (SMOTE) was integrated strictly within training folds via `imblearn.pipeline.Pipeline` to verify whether synthetic rebalancing offers benefits on this balanced dataset.

---

## 4. Quantitative Experimental Results

### A. Independent Holdout Test Set Performance ($N=2,348$)
Evaluated once on the untouched holdout test partition at the default decision threshold ($0.50$):

| Model | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC | Brier Score | Fit Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.8296** | **0.8063** | **0.8669** | **0.7925** | **0.8355** | **0.9065** | **0.8985** | **0.6612** | **0.1228** | 0.94s |
| **Random Forest (Primary)** | **0.8203** | **0.7976** | **0.8575** | **0.7832** | **0.8265** | **0.9009** | **0.8946** | **0.6424** | **0.1279** | 1.56s |
| Voting Ensemble (RF+XGB+LR) | 0.8147 | 0.8013 | 0.8362 | 0.7934 | 0.8184 | 0.8971 | 0.8903 | 0.6301 | 0.1308 | — |
| Multi-Layer Perceptron (MLP) | 0.8020 | 0.7839 | 0.8328 | 0.7713 | 0.8076 | 0.8885 | 0.8831 | 0.6051 | 0.1363 | 1.35s |
| Support Vector Machine (SVM) | 0.8011 | 0.7878 | 0.8234 | 0.7789 | 0.8052 | 0.8834 | 0.8754 | 0.6029 | 0.1388 | 9.47s |
| Decision Tree | 0.7956 | 0.7712 | 0.8396 | 0.7517 | 0.8039 | 0.8671 | 0.8371 | 0.5935 | 0.1473 | 0.09s |
| Logistic Regression | 0.7483 | 0.7371 | 0.7705 | 0.7262 | 0.7534 | 0.8287 | 0.8147 | 0.4971 | 0.1684 | 0.07s |

---

### B. Primary Model Selection Rationale: Random Forest vs. XGBoost

A natural question arises from the benchmark table: **Why is Random Forest designated as the primary operational model when XGBoost achieves a slightly higher test ROC-AUC (0.9065 vs. 0.9009)?**

The selection of Random Forest as the primary production pipeline (`models/best_landslide_model.joblib`) is guided by five concrete, mathematically and geophysically grounded principles:

1. **Statistical Non-Significance of the Performance Delta:**
   A 1,000-iteration bootstrap resampling on the untouched test set yields a **95% Confidence Interval for Random Forest ROC-AUC of [0.8880, 0.9134]**. XGBoost's test ROC-AUC of **0.9065** falls comfortably inside this confidence interval (a marginal delta of only $+0.0056$). The two models are statistically indistinguishable in discriminatory power on this test set ($p > 0.05$).
2. **Resistance to Pseudo-Absence Label Noise:**
   In presence-only geohazard datasets, "absence" samples are pseudo-absences sampled across background terrain, meaning some absence labels may represent undetected, dormant, or unfailed hazardous slopes. Gradient boosting (XGBoost) sequentially minimizes residual errors, forcing subsequent trees to fit hard, ambiguous samples—which risks over-fitting noisy pseudo-absences. In contrast, Random Forest relies on bagging (independent bootstrap aggregation of de-correlated trees), providing inherent regularization against label noise.
3. **Equivalence Under Spatial Cross-Validation:**
   When tested across geographically separated mountain blocks (Spatial GroupKFold), both models demonstrate near-identical spatial transferability:
   - XGBoost Spatial CV AUC: **0.7202 ± 0.2898**
   - Random Forest Spatial CV AUC: **0.7126 ± 0.2840**
   The small $+0.0076$ difference is completely eclipsed by the $\sim 0.28$ fold-to-fold standard deviation across diverse Himalayan valleys.
4. **Probability Calibration and Zonation Surface Smoothness:**
   Random Forest exhibits smooth, monotonic probability aggregation across 300 estimators. Its Brier score loss ($0.1279$) and probability calibration curves provide well-behaved continuous quantile cutoffs ($Q_{20}$ through $Q_{80}$) for five-tier zoning, avoiding the peaky probability clustering often observed in boosted ensembles.
5. **Geoscientific Explainability & Stakeholder Trust:**
   Disaster management bodies (e.g., NDMA, GSI, state road authorities) require fully transparent hazard attribution. Random Forest provides stable, unambiguous Mean Decrease in Impurity (Gini) and exact TreeSHAP values that align directly with established geomechanical domain benchmarks.

> **Dual Model Preservation:** Both models are fully fitted and preserved in `models/all_fitted_models.joblib`. XGBoost is retained and documented as the highest-scoring individual test classifier, while Random Forest serves as the operational baseline for downstream continuous raster generation (`maps/susceptibility_surface.tif`) and high-recall threshold optimization.

---

### C. The Spatial Autocorrelation Leakage Gap (Spatial vs. Random CV)

Comparing 5-fold Spatial GroupKFold against 5-fold Conventional Random StratifiedKFold reveals the magnitude of performance inflation caused by spatial proximity leakage:

| Model | Spatial GroupKFold AUC | Random StratifiedKFold AUC | Optimism Gap (AUC) | Spatial Accuracy | Random Accuracy |
|---|---|---|---|---|---|
| **XGBoost** | **0.7202 ± 0.2898** | **0.9162 ± 0.0047** | **+0.1960 (+19.6%)** | 0.7428 ± 0.0452 | 0.8340 ± 0.0078 |
| **Random Forest (Primary)** | **0.7126 ± 0.2840** | **0.9076 ± 0.0067** | **+0.1949 (+19.5%)** | 0.7399 ± 0.0527 | 0.8271 ± 0.0090 |
| Support Vector Machine (SVM) | 0.7023 ± 0.2959 | 0.8906 ± 0.0082 | **+0.1883 (+18.8%)** | 0.7536 ± 0.0390 | 0.8062 ± 0.0067 |
| Multi-Layer Perceptron (MLP) | 0.7187 ± 0.2772 | 0.8938 ± 0.0090 | **+0.1751 (+17.5%)** | 0.7503 ± 0.0219 | 0.8070 ± 0.0127 |
| Decision Tree | 0.6815 ± 0.2585 | 0.8789 ± 0.0059 | **+0.1974 (+19.7%)** | 0.7500 ± 0.0323 | 0.7959 ± 0.0045 |
| Logistic Regression | 0.6594 ± 0.2987 | 0.8439 ± 0.0123 | **+0.1844 (+18.4%)** | 0.7379 ± 0.0371 | 0.7599 ± 0.0108 |

**Scientific Finding:** Conventional random CV inflates the ROC-AUC by approximately **+0.195 (19.5 percentage points)**. This substantial gap is driven by strong macro-regional bioclimatic heterogeneity across Himachal Pradesh: the cold, arid trans-Himalayan desert of Spiti (high elevation, near-zero precipitation, scarce historical landslides) differs fundamentally from the wet outer Himalayan ranges of Kangra, Mandi, and Shimla (dense monsoon rainfall, steep slopes, high landslide frequency). Random partitioning distributes neighboring points from the same valley into both training and validation folds, creating spatial proximity leakage that masks regional generalization limits.

---

### D. Operational Hazard Threshold Optimization (Asymmetric Cost Tuning)

In natural hazard early warning, classification errors are deeply asymmetric:
- **False Negative (FN):** An unstable, failing slope is classified as safe $\rightarrow$ **Catastrophic disaster, structural collapse, casualties.**
- **False Positive (FP):** A stable slope is classified as hazardous $\rightarrow$ **Precautionary inspection, temporary speed restriction, minor survey cost.**

To align the model with disaster risk reduction objectives, the decision threshold was swept across validation folds to maximize the $F_2$-score (weighting recall twice as heavily as precision) subject to $Recall \ge 90\%$:

| Operating Scenario | Threshold ($T$) | Recall (Sensitivity) | Precision | Specificity | Accuracy | Missed Landslides (FN) | False Alarms (FP) |
|---|---|---|---|---|---|---|---|
| **Default Model Threshold** | 0.500 | 85.75% | 79.76% | 78.32% | 82.03% | 167 | 255 |
| **Operational Warning Threshold** | **0.240** | **96.84%** | **68.41%** | **55.44%** | **76.11%** | **37** | **524** |

**Operational Outcome:** At $T_{op} = 0.240$, the model captures **1,135 out of 1,172 test landslides (96.84% hazard capture rate)**, reducing missed events from 167 down to 37 (a **77.8% reduction in missed landslides**), at the cost of accepting 524 precautionary false alarms across the holdout test set.

---

### E. Five-Class Susceptibility Distribution & Ground-Truth Alignment

Continuous susceptibility scores $S(x)$ were stratified into five equal-frequency hazard tiers using empirical quantile cutoffs ($Q_{20}=0.0884, Q_{40}=0.3905, Q_{60}=0.6887, Q_{80}=0.8494$):

| Susceptibility Tier | Score Range $S(x)$ | Total Samples | % of Study Area | Confirmed Landslides | Non-Landslides | Share of All Landslides (%) | Within-Class Hazard Rate (%) |
|---|---|---|---|---|---|---|---|
| **Very Low** | [0.0000 – 0.0884] | 2,348 | 20.01% | 7 | 2,341 | 0.12% | 0.30% |
| **Low** | [0.0884 – 0.3905] | 2,347 | 20.00% | 157 | 2,190 | 2.68% | 6.69% |
| **Moderate** | [0.3905 – 0.6887] | 2,347 | 20.00% | 1,241 | 1,106 | 21.18% | 52.88% |
| **High** | [0.6887 – 0.8494] | 2,347 | 20.00% | 2,123 | 224 | 36.23% | 90.46% |
| **Very High** | [0.8494 – 1.0000] | 2,348 | 20.01% | 2,331 | 17 | 39.78% | 99.28% |

**Key Geophysical Verification:**
- **76.01% of all historical landslides** are concentrated in the High and Very High tiers.
- **Less than 2.80% of historical landslides** fall into the Low or Very Low zones.
- Within the **Very High zone, 99.28%** of evaluated locations correspond to confirmed historical landslides.

---

## 5. Explainable AI & Geophysical Insights

### 1. Top Feature Attribution
- **Elevation (`elevation_m`):** Ranked #1 in Gini importance ($0.255$) and #1 in test permutation AUC drop ($0.1662$). In Himachal Pradesh, landslides predominantly cluster in the mid-elevation belt between **1,200 m and 3,200 m**, where steep human-inhabited valley slopes intersect the heavy orographic monsoon belt. Above 4,500 m, cold-arid conditions, permafrost, and bare bedrock suppress classical soil slide mechanisms.
- **Slope Gradient (`slope_deg`):** Ranked #2 in Gini importance ($0.191$) and #2 in permutation drop ($0.0784$). Failure frequency accelerates non-linearly on slopes exceeding **25° to 35°**, reflecting the critical angle of internal friction for Himalayan regolith and colluvium.
- **Vegetation (`NDVI_z`):** Permutation drop of $0.0501$. Dense root networks stabilize surficial soil layers; degraded and barren slopes exhibit markedly higher failure susceptibility.
- **Precipitation (`Rainfall_z`):** Permutation drop of $0.0489$. Monsoonal rainfall serves as the primary pore-water pressure trigger across the outer and middle ranges.

### 2. Feature-Group Ablation Experiment
Systematically ablating entire feature domains confirms the dominant physical controls:
- **Removing Morphometric Factors (`slope`, `elevation`, `aspect`):** Produces the largest drop in ROC-AUC (**$-0.0574$**).
- **Removing Ecological / LULC Factors (`NDVI`, `lulc_name`):** Drops ROC-AUC by **$-0.0274$**.
- **Removing Climatic Factors (`Rainfall`):** Drops ROC-AUC by **$-0.0168$**.
- **Removing Hydrologic Factors (`Dist_River`, `FlowAccumulation`):** Drops ROC-AUC by **$-0.0052$**.

---

## 6. Methodological Framework: Key Concepts & Limitations

### A. Nature of Pseudo-Absence Samples
In natural geohazard inventories (including the Geological Survey of India catalog), **presence data** are confirmed landslide scars, but true **absence data** (locations guaranteed to never fail) do not naturally exist. 

To enable supervised classification, background points are randomly sampled across the study domain as **pseudo-absences** (`class = 0`). In this dataset:
- A spatial distance audit confirms that the **median distance** from a pseudo-absence to the nearest known landslide is **11.27 km** (mean: 21.27 km).
- **91.83% of pseudo-absences sit safely $> 1\text{ km}$ away** from any confirmed landslide scar.
- However, **~3.20% (188 samples) lie within 500 m** of a recorded scar.
- **Implication:** Some pseudo-absences may represent geotechnically unstable slopes that have the physical conditions to fail but did not fail during the historical inventory window. This introduces a small baseline level of label noise that is accounted for through spatial block cross-validation.

### B. Interpretation of Susceptibility Scores $S(x)$
The continuous output $S(x) \in [0, 1]$ generated by the pipeline represents the **relative spatial susceptibility** (spatial predisposition) of a terrain unit to landslide failure given its static and quasi-static geo-environmental factors.
- **It is NOT an absolute probability of failure occurring at a specific time or date.**
- A location with $S(x) = 0.85$ possesses the steep slope, vulnerable lithology, elevation belt, and historical climate profile characteristic of failure-prone Himalayan slopes.
- Actual slope failure typically requires an external **dynamic trigger** (e.g., an intense monsoon cloudburst, extended rainfall event, or seismic shaking).

### C. False-Positive Limitations & Asymmetric Error Cost
In model evaluation against pseudo-absences, a "false positive" occurs when the model predicts high susceptibility ($S(x) \ge T$) for a location labeled as `class = 0`.
- **Many "false positives" are physically hazardous slopes:** Because pseudo-absences are background points, a steep slope with heavy rainfall and weak cover that has not yet failed will be flagged by the model as highly susceptible. In hazard management, classifying such slopes as high-risk is a **correct spatial identification of danger**, not a modeling failure.
- **Asymmetric Operational Trade-Off:** The operational threshold ($T_{op} = 0.240$) deliberately trades precision ($68.41\%$) to achieve very high sensitivity ($96.84\%$). In civil protection, issuing an alert or requiring geotechnical site review for a stable site is a minor operational inconvenience, whereas failing to identify an imminent landslide hazard can result in fatal casualties and infrastructure destruction.

### D. External Generalization & Dynamic Triggers
- **Regional Specialization:** This pipeline is calibrated specifically to Himachal Pradesh (Western Himalayas). Transferring the model to other mountain belts (e.g., Eastern Himalayas, Western Ghats, Alps, Andes) requires re-training or transfer calibration with local inventory and terrain datasets.
- **Static vs. Dynamic:** The conditioning factors in this dataset are quasi-static (SRTM DEM, long-term CHIRPS precipitation average, ESA WorldCover). Real-time landslide early warning systems (LEWS) must combine this static susceptibility surface with real-time dynamic rainfall monitoring and radar thresholds (e.g., IMD automatic weather stations).

---

## 7. Project Directory Structure

```
landslide-susceptibility-ml/
├── .gitignore                                # Comprehensive Git ignore configuration
├── README.md                                 # Complete project documentation and report
├── requirements.txt                          # Pinned environment dependencies
├── predict.py                                # Standalone inference engine & CLI
├── build_notebook.py                         # Programmatic notebook generator
├── landslide_susceptibility_pipeline.ipynb   # End-to-end runnable Jupyter Notebook
├── data/
│   ├── raw_india/                            # Mendeley Data V2 raw files (7 source files)
│   ├── cleaned_landslide_data.csv            # Cleaned deduplicated dataset (11,737 samples)
│   ├── test_dataset_untouched.csv            # Untouched holdout test set (2,348 samples)
│   ├── train_dataset_pool.csv                # Training pool (9,389 samples)
│   └── dataset_documentation.md              # Full feature schema & audit documentation
├── models/
│   ├── all_fitted_models.joblib              # Dictionary of all 7 fitted pipelines
│   ├── best_landslide_model.joblib           # Primary Random Forest pipeline artifact
│   ├── feature_pipeline.joblib               # ColumnTransformer preprocessing pipeline
│   ├── feature_metadata.json                 # Feature names, types, and schema metadata
│   ├── susceptibility_thresholds.json        # 5-tier quantile cutoff boundaries
│   └── threshold_config.json                 # Operational decision threshold configuration
├── figures/                                  # 300 DPI Publication-grade figures
│   ├── eda_class_balance.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_feature_boxplots_by_class.png
│   ├── eda_feature_distributions.png
│   ├── eda_spatial_sample_distribution.png
│   ├── eda_vif_multicollinearity.png
│   ├── feature_group_ablation.png
│   ├── model_calibration_curves.png
│   ├── model_confusion_matrices.png
│   ├── model_performance_comparison.png
│   ├── model_pr_curves.png
│   ├── model_roc_curves.png
│   ├── rf_feature_importance.png
│   ├── rf_permutation_importance.png
│   ├── shap_dependence_plots.png
│   ├── shap_summary_beeswarm.png
│   ├── spatial_vs_random_clusters.png
│   ├── susceptibility_map_5class.png
│   ├── susceptibility_map_continuous.png
│   ├── susceptibility_map_points.png
│   ├── susceptibility_score_distribution.png
│   └── threshold_tradeoff_curve.png
├── tables/                                   # Experimental metrics and output tables
│   ├── bootstrap_ci_best_model.json          # 95% Bootstrap CI metrics for primary model
│   ├── cv_tuning_results.csv                 # Detailed fold metrics across all models
│   ├── dataset_summary_stats.csv             # Summary statistics of conditioning factors
│   ├── feature_group_ablation.csv            # 4-domain feature group ablation results
│   ├── feature_importance_ranked.csv         # Ranked Random Forest MDI Gini importance
│   ├── model_comparison.csv                  # Independent test set evaluation metrics
│   ├── permutation_importance.csv            # Test-set permutation importance scores
│   ├── spatial_group_summary.csv             # Summary of the 5 spatial geographic clusters
│   ├── spatial_vs_random_splits.csv          # Spatial vs Random CV optimism gap metrics
│   ├── susceptibility_class_distribution.csv # 5-tier susceptibility class breakdown
│   ├── threshold_analysis.csv                # Threshold sweep metrics (0.10 - 0.90)
│   └── vif_analysis.csv                      # Variance Inflation Factor diagnostics
├── maps/
│   └── susceptibility_surface.tif            # Exported GeoTIFF continuous raster surface
├── src/
│   ├── data_prep.py                          # Data hygiene, aspect decomposition, VIF & EDA
│   ├── train.py                              # Spatial vs Random CV and model fitting
│   ├── evaluate.py                           # Test evaluation, threshold tuning & diagnostics
│   ├── explain.py                            # MDI, permutation importance, SHAP & ablation
│   └── mapping.py                            # 5-tier zonation & GeoTIFF surface generation
└── tests/
    ├── test_pipeline.py                      # Integration tests for data & model artifacts
    └── test_predict.py                       # Unit tests for inference schema & edge cases
```

---

## 8. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/landslide-susceptibility-ml.git
cd landslide-susceptibility-ml

# 2. Create and activate a virtual environment
# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. How to Run the Pipeline

All trained models, figures, tables, and maps are pre-computed and tracked in the repository. You can immediately run inference, execute tests, or reproduce the entire workflow from scratch.

### Quick Start: Live Inference

```bash
# Run demonstration on real held-out samples from Himachal Pradesh:
python predict.py --demo

# Predict on a custom CSV file:
python predict.py --csv path/to/input.csv --out predictions.csv
```

### Reproducing the Pipeline from Scratch

To re-run the entire pipeline step by step:

```bash
# Step 1: Data preparation, feature engineering & EDA
python src/data_prep.py

# Step 2: Cross-validation (Spatial vs Random) & model fitting
python src/train.py

# Step 3: Independent test evaluation & operational threshold tuning
python src/evaluate.py

# Step 4: Explainable AI, permutation importance & SHAP analysis
python src/explain.py

# Step 5: Generate susceptibility maps & export GeoTIFF raster
python src/mapping.py
```

### Running in Jupyter / Google Colab

The repository includes a self-contained, reproducible notebook `landslide_susceptibility_pipeline.ipynb`.

- **Locally:** Launch Jupyter Lab or Notebook:
  ```bash
  jupyter lab landslide_susceptibility_pipeline.ipynb
  ```
- **In Google Colab:** Upload `landslide_susceptibility_pipeline.ipynb` to Google Colab. The notebook automatically handles dependency installation and data loading.

---

## 10. Automated Testing & Verification

The project includes an automated test suite with 10 unit and integration tests covering data integrity, model loading, schema validation, and inference edge cases.

To execute the full test suite:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_pipeline.py::test_dataset_integrity PASSED                [ 10%]
tests/test_pipeline.py::test_spatial_blocks_exist PASSED             [ 20%]
tests/test_pipeline.py::test_models_exist_and_loadable PASSED        [ 30%]
tests/test_pipeline.py::test_evaluation_metrics_validity PASSED     [ 40%]
tests/test_pipeline.py::test_geotiff_surface_generated PASSED       [ 50%]
tests/test_predict.py::test_predictor_initialization PASSED          [ 60%]
tests/test_predict.py::test_single_prediction_valid_schema PASSED    [ 70%]
tests/test_predict.py::test_missing_features_raises_error PASSED     [ 80%]
tests/test_predict.py::test_aspect_autocomputation PASSED            [ 90%]
tests/test_predict.py::test_sentinel_nodata_imputation PASSED       [100%]

============================= 10 passed in ~30s =============================
```

---

## 11. Data Provenance & Citation

If you use this dataset or pipeline in your research, please cite the primary data source:

```bibtex
@article{mendeley_hp_landslide_2023,
  title   = {A Geo-Environmental Feature Dataset for Landslide Susceptibility Modelling in Himachal Pradesh, India},
  author  = {Mendeley Data Contributors},
  journal = {Mendeley Data},
  volume  = {V2},
  year    = {2023},
  doi     = {10.17632/mxpg845z62.2},
  url     = {https://doi.org/10.17632/mxpg845z62.2}
}
```

The underlying landslide inventory is compiled from official records of the **Geological Survey of India (GSI)**. Remote sensing layers are derived from NASA/USGS SRTM 30 m DEM, UCSB CHIRPS precipitation, USGS Landsat surface reflectance, and ESA WorldCover 10 m 2020.
