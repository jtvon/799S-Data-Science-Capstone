# 799S Data Science Capstone

A semester-long integrated capstone project applying machine learning and data science techniques to insurance fraud detection and vehicle damage classification. Each week's notebook applies that week's course concepts to the project datasets, building toward two milestone deliverables.

---

## Project Overview

The capstone has two primary problem domains:

1. **Insurance Fraud Detection** — Binary classification on tabular claims data to predict whether a claim is fraudulent.
2. **Vehicle Damage Classification** — Binary classification using vehicle images and tabular policy data to predict whether a vehicle is damaged and estimate claim amounts.

---

## Datasets

### Tabular Datasets (`data/`)

| File | Records | Features | Target | Fraud Rate |
|------|---------|----------|--------|------------|
| `carclaims_cleaned.csv` | 15,100 | 31 | `FraudFound` (0/1) | ~6% |
| `insurance_claims_cleaned.csv` | ~1,000 | 35 | `fraud_reported` (0/1) | ~25% |
| `insurance_fraud_data.csv` | 12,002 | 29 | `fraud reported` (Y/N) | ~33% |

The car claims dataset (`carclaims.csv`) was cleaned by removing records with driver age < 16 and dropping the `PolicyNumber` and `AgeOfPolicyHolder` columns.

### Image Dataset (`Fast_Furious_Insured/`)

The *Fast, Furious & Insured* dataset contains vehicle photos with associated insurance policy metadata. The prediction task is to classify vehicle condition and estimate claim payout.

| File | Description |
|------|-------------|
| `train.csv` | 1,399 labeled records (image path, insurance company, cost of vehicle, coverage, expiry date, condition, claim amount) |
| `test.csv` | Unlabeled records for prediction |
| `sample_submission.csv` | Submission format |
| `trainImages/` | Training vehicle images (JPG) |
| `testImages/` | Test vehicle images (JPG) |

**Target distribution** (`Condition`): ~93% damaged (1), ~7% not damaged (0) — highly imbalanced.

---

## Repository Structure

```
799S-Data-Science-Capstone/
├── data/
│   ├── carclaims_cleaned.csv
│   ├── insurance_claims_cleaned.csv
│   └── insurance_fraud_data.csv
├── Fast_Furious_Insured/
│   ├── train.csv / test.csv / sample_submission.csv
│   ├── trainImages/
│   └── testImages/
├── Capstone_Notebooks/
│   ├── Car_claims_eda.ipynb          # EDA: car claims fraud dataset
│   ├── Insurance_claims_eda.ipynb    # EDA: insurance claims dataset
│   ├── Insurance_Fraud_2.ipynb       # EDA: insurance_fraud_data dataset
│   ├── FFI_eda.ipynb                 # CNN image classification (vehicle damage)
│   ├── NN_CarClaims_Classification.ipynb  # Neural network fraud classifiers
│   └── Week-1-jupyter-notebook.ipynb
├── Week-1-jupyter-notebook.ipynb     # Linear regression, polynomial terms, VIF
├── Week-2-jupyter-notebook.ipynb     # Ridge, Lasso, ElasticNet
├── Week-3-jupyter-notebook.ipynb     # Forward/backward selection, PCR, PLSR
├── Week-4-jupyter-notebook.ipynb     # Logistic regression, feature scaling
├── Week-5-jupyter-notebook.ipynb     # SVM, kernel trick, regularization
├── Week-6-jupyter-notebook.ipynb     # Decision trees, Random Forests
├── Week-8-jupyter-notebook.ipynb     # K-Nearest Neighbors, distance metrics
├── Week-9-jupyter-notebook.ipynb     # Gradient Boosting (learning rate, depth, estimators)
├── Week-10-jupyter-notebook.ipynb    # K-Means clustering, elbow method, silhouette score
├── Week-11-jupyter-notebook.ipynb    # DBSCAN, HAC, linkage methods, dendrograms
├── Week-13-jupyter-notebook.ipynb    # Bernoulli, Gaussian, and Multinomial Naive Bayes
└── Week-14-jupyter-notebook.ipynb    # Gaussian Mixture Models
```

---

## Weekly Notebooks Summary

| Week | Topic | Key Concepts |
|------|-------|-------------|
| 1 | Regression foundations | Linear regression, polynomial/interaction terms, multicollinearity, VIF |
| 2 | Regularized regression | Ridge, Lasso, ElasticNet with cross-validated alpha search |
| 3 | Feature selection & dimensionality reduction | Forward/backward selection, PCR, PLSR |
| 4 | Classification baseline | Logistic regression, StandardScaler vs MinMaxScaler, threshold tuning |
| 5 | Support Vector Machines | SVC with linear/RBF/poly/sigmoid kernels, C & gamma tuning |
| 6 | Ensemble methods | Decision trees, Random Forests, GridSearchCV, feature importance |
| 8 | Instance-based learning | K-Nearest Neighbors, p-norm distance metrics |
| 9 | Boosting | Gradient Boosting Classifier, learning rate, n_estimators, max_depth |
| 10 | Unsupervised clustering | K-Means, elbow method, silhouette score |
| 11 | Density-based & hierarchical clustering | DBSCAN, HAC (single/complete/average/ward), dendrograms |
| 13 | Probabilistic classifiers | Bernoulli NB, Gaussian NB, Multinomial NB; SMOTE resampling |
| 14 | Generative models | Gaussian Mixture Models (GMM) |

---

## Capstone Notebooks

### `Capstone_Notebooks/FFI_eda.ipynb`
CNN-based binary image classifier for vehicle damage detection using the Fast, Furious & Insured dataset. Builds and compares multiple architectures: baseline CNN → regularized CNN → advanced architectures. Uses TensorFlow/Keras with early stopping (monitored on `val_loss`) and 80/10/10 train/val/test splits.

### `Capstone_Notebooks/NN_CarClaims_Classification.ipynb`
Fully connected neural network classifiers for fraud detection on the car claims and insurance claims datasets. Experiments with activation functions, learning rates, and dropout rates. Optimizes for **recall** (minimizing false negatives) using early stopping on `val_recall` with balanced class weights.

### `Capstone_Notebooks/Car_claims_eda.ipynb` / `Insurance_claims_eda.ipynb` / `Insurance_Fraud_2.ipynb`
Exploratory data analysis including missingness profiling, outlier detection (IQR + robust Z-score), feature engineering, PCA dimensionality reduction, and fraud rate visualizations by feature.

---

## Key Technical Notes

- **Class imbalance**: All three fraud datasets are heavily imbalanced. Strategies used include `class_weight='balanced'`, SMOTE oversampling, and threshold tuning on predicted probabilities.
- **Preprocessing**: Consistent `encoding()` utility for label and one-hot encoding; `StandardScaler` and `MinMaxScaler` applied on training folds only to prevent data leakage.
- **Evaluation metric**: Classification notebooks optimize for **recall** (fraud detection priority) using `GridSearchCV` with `scoring='recall'` and `RepeatedStratifiedKFold`.
- **Image pipeline**: Vehicle images loaded as 150×150 RGB tensors, normalized to [0, 1]. Imbalanced classes handled via class weights in CNN training.

---

## Libraries & Dependencies

- **Data**: `pandas`, `numpy`
- **ML**: `scikit-learn`, `imbalanced-learn`
- **Deep Learning**: `TensorFlow` / `Keras`
- **Visualization**: `matplotlib`, `seaborn`, `scipy`
