#  Heart Disease Prediction using Machine Learning

> **A complete end-to-end Machine Learning pipeline to predict the presence of heart disease in patients using clinical attributes — with model comparison, hyperparameter tuning, and feature importance analysis.**


##  Problem Statement

Heart disease is one of the leading causes of death globally. Early and accurate prediction can save lives by enabling timely medical intervention. This project builds a **binary classification model** to predict whether a patient has heart disease (1) or not (0) based on 13 clinical features.

---

##  Dataset Overview

| Property | Details |
|---|---|
| **File** | `dataset_heart.csv` |
| **Rows** | 270 |
| **Columns** | 14 |
| **Target** | `heart disease` (0 = No Disease, 1 = Disease) |
| **Missing Values** |  None |
| **Duplicates** |  None |

###  Features

| # | Feature | Description |
|---|---|---|
| 1 | `age` | Age of the patient |
| 2 | `sex` | Sex (1 = Male, 0 = Female) |
| 3 | `chest pain type` | Type of chest pain (1–4) |
| 4 | `resting blood pressure` | Resting BP in mm Hg |
| 5 | `serum cholesterol` | Serum cholesterol in mg/dl |
| 6 | `fasting blood sugar` | >120 mg/dl (1 = True, 0 = False) |
| 7 | `resting electrocardiographic results` | ECG results (0, 1, 2) |
| 8 | `max heart rate` | Maximum heart rate achieved |
| 9 | `exercise induced angina` | Angina from exercise (1 = Yes, 0 = No) |
| 10 | `oldpeak` | ST depression induced by exercise |
| 11 | `ST segment` | Slope of peak exercise ST segment |
| 12 | `major vessels` | Number of major vessels (0–3) colored by flourosopy |
| 13 | `thal` | Thalassemia type (3 = Normal, 6 = Fixed defect, 7 = Reversable defect) |
| 14 | `heart disease` | **Target** — 0 = No, 1 = Yes |

---

##  Project Workflow

```
 Load Data
    ↓
 Exploratory Data Analysis (EDA)
    ↓
 Data Preprocessing
  → Target encoding (1→0, 2→1)
  → Train-Test Split (80/20)
  → Feature Scaling (StandardScaler)
  → Handle class imbalance (SMOTE)
    ↓
 Model Training + Hyperparameter Tuning (GridSearchCV)
  → Logistic Regression
  → Support Vector Classifier
  → Random Forest Classifier
  → XGBoost Classifier
    ↓
 Model Evaluation
  → ROC-AUC Curve Comparison
  → Confusion Matrix
  → Classification Report
    ↓
 Best Model Selection → Logistic Regression (AUC = 0.93)
    ↓
 Feature Importance Analysis
```

---

##  Exploratory Data Analysis

- **Class Distribution:** 150 patients without heart disease vs 120 with — slightly imbalanced
- **Correlation Analysis:** `thal`, `chest pain type`, and `max heart rate` showed the strongest correlation with the target
- **No missing or duplicate values** were found — the dataset was clean and ready for modeling

---

##  Models Used

All models were tuned using **GridSearchCV with 5-fold cross-validation**, optimized for **ROC-AUC score**.

| Model | Best Parameters |
|---|---|
| Logistic Regression | C=0.01, penalty='l2', solver='lbfgs' |
| Support Vector Classifier | C=0.01, kernel='linear', gamma='scale' |
| Random Forest Classifier | max_depth=5, min_samples_split=5, n_estimators=200 |
| XGBoost Classifier | learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.8 |

**SMOTE (Synthetic Minority Over-sampling Technique)** was applied to balance the training data from (117 vs 99) → **(117 vs 117)** before model training.

---

##  Results

###  ROC-AUC Score Comparison

| Model | ROC-AUC Score |
|---|---|
|  **Logistic Regression** | **0.93** |
| Support Vector Classifier | 0.93 |
| Random Forest Classifier | 0.92 |
| XGBoost Classifier | 0.91 |

###  Best Model — Logistic Regression: Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (No Disease) | 0.86 | 0.97 | 0.91 | 33 |
| 1 (Disease) | 0.94 | 0.76 | 0.84 | 21 |
| **Accuracy** | | | **0.89** | **54** |
| Macro Avg | 0.90 | 0.87 | 0.88 | 54 |
| Weighted Avg | 0.89 | 0.89 | 0.89 | 54 |

###  Confusion Matrix

```
              Predicted 0    Predicted 1
Actual 0         32              1
Actual 1          5             16
```

 32 patients correctly predicted as **No Disease**
 16 patients correctly predicted as **Has Disease**
 Only 5 false negatives (disease missed) — critical metric in healthcare

---

##  Key Findings

1. **Best Model:** Logistic Regression achieved the highest ROC-AUC of **0.93**
2. **Top Predictive Features:** `thal`, `major vessels`, and `chest pain type` were the most important features for predicting heart disease
3. **SMOTE significantly improved** model performance on the minority class (heart disease = 1)
4. **High Recall for Class 0 (0.97):** The model is very good at correctly identifying patients who do NOT have heart disease
5. The model achieves an overall **89% accuracy** on unseen test data

---

##  Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots and heatmaps |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `imbalanced-learn` | SMOTE for class balancing |
| `xgboost` | Gradient boosting classifier |

---

