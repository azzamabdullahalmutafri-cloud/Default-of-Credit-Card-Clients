# Default of Credit Card Clients — ML Classification

A machine learning project that predicts whether a credit card client will default on their next payment, using the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

---

## 📋 Project Overview

This notebook covers the full ML pipeline — from exploratory data analysis (EDA) to model training and evaluation — on a real-world binary classification problem. The target variable is whether a client will **default on their credit card payment next month** (1 = default, 0 = no default).

---

## 📦 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Records:** 30,000 credit card clients in Taiwan (April–September 2005)
- **Features:** 24 input features + 1 target variable
- **Class distribution:** ~78% non-default, ~22% default (imbalanced)
- **Missing values:** None
- **Duplicate rows:** None

### Feature Summary

| Feature | Description |
|---|---|
| `LIMIT_BAL` | Credit limit (NT dollar) |
| `SEX` | Gender (1 = male, 2 = female) |
| `EDUCATION` | Education level (1 = grad school, 2 = university, 3 = high school, etc.) |
| `MARRIAGE` | Marital status (1 = married, 2 = single, 3 = other) |
| `AGE` | Age in years |
| `PAY_0` – `PAY_6` | Repayment status for September–April (-1 = paid on time, 1–8 = months delayed) |
| `BILL_AMT1` – `BILL_AMT6` | Bill statement amounts (September–April) |
| `PAY_AMT1` – `PAY_AMT6` | Previous payment amounts (September–April) |
| `default payment next month` | Target variable (1 = default, 0 = no default) |

---

## 🔍 Workflow

### 1. Exploratory Data Analysis (EDA)
- Dataset shape, data types, missing value checks (no missing values found)
- Duplicate detection (no duplicates found)
- Outlier detection using IQR method per column
- Descriptive statistics and class distribution analysis
- Categorical feature value counts for SEX, EDUCATION, MARRIAGE, and target

### 2. Preprocessing & Feature Engineering

Three engineered features were created:

| Feature | Description |
|---|---|
| `utilization_avg` | Average credit utilization ratio (avg bill / credit limit) |
| `avg_payment_delay` | Average payment delay status across 6 months |
| `payment_ratio` | Ratio of last payment amount to last bill amount |

Additional preprocessing steps:
- Outlier capping for `EDUCATION` (values > 4 or 0 → mapped to 4) and `MARRIAGE` (value 0 → mapped to 3)
- One-hot encoding for `EDUCATION` and `MARRIAGE` using `pd.get_dummies`
- Infinite and NaN values replaced with column means
- 80/20 stratified train/test split
- `StandardScaler` applied for distance-based and neural network models
- Class imbalance handled via `scale_pos_weight`

### 3. Models Trained

| Model | Evaluation Method |
|---|---|
| Logistic Regression | Test Set (80/20 split) |
| K-Nearest Neighbors (KNN, k=7) | Test Set (80/20 split) |
| MLP Neural Network | Test Set → then 10-Fold Stratified CV |
| XGBoost (initial) | Test Set (80/20 split) |
| XGBoost (RandomizedSearchCV, 50 iterations) | 10-Fold Stratified CV |
| XGBoost (GridSearchCV tuned) | 10-Fold Stratified CV |
| CatBoost (GridSearchCV tuned) | 10-Fold Stratified CV |

### 4. Evaluation Strategy
- **Baseline models:** held-out test set (20% of data, 6,000 records)
- **Tuned models:** 10-Fold Stratified Cross-Validation with out-of-fold predictions
- Metrics: Accuracy, ROC-AUC, Precision, Recall, F1-Score
- Custom probability threshold (0.3) used to improve recall on the minority (default) class

---

## 📊 Results

Results are grouped by evaluation method. Baseline models were evaluated on a held-out test set (20%), while the final tuned models used 10-Fold Stratified Cross-Validation.

### Baseline Models (Test Set — 6,000 records)

| Model | Accuracy | ROC-AUC | F1 (Default Class) |
|---|---|---|---|
| Logistic Regression | 68.1% | 0.712 | 0.46 |
| KNN (k=7) | 80.2% | 0.713 | 0.43 |
| MLP Neural Network | 78.8% | 0.718 | 0.47 |
| XGBoost (initial) | 67.9% | 0.775 | 0.50 |
| XGBoost (RandomizedSearch tuned) | 66.0% | 0.780 | 0.50 |

### Tuned Models (10-Fold Stratified Cross-Validation)

| Model | Accuracy | ROC-AUC | F1 (Default Class) |
|---|---|---|---|
| **CatBoost** ⭐ | **76.5%** | **0.787** | **0.54** |
| XGBoost | 76.2% | 0.786 | 0.54 |
| MLP Neural Network | 77.3% | 0.709 | 0.43 |

> **CatBoost** achieved the best overall ROC-AUC of **0.787**. While KNN shows the highest raw accuracy (80.2%), this is misleading due to class imbalance — CatBoost and XGBoost significantly outperform on the minority (default) class as measured by ROC-AUC and F1-score.

### Best CatBoost Hyperparameters (GridSearchCV)

| Parameter | Value |
|---|---|
| `max_depth` | 7 |
| `learning_rate` | 0.03 |
| `n_estimators` | 400 |
| `subsample` | 1.0 |

### Best XGBoost Hyperparameters (GridSearchCV)

| Parameter | Value |
|---|---|
| `max_depth` | 5 |
| `learning_rate` | 0.03 |
| `n_estimators` | 200 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

---

## 🛠️ Requirements

```bash
pip install catboost xgboost scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

Or if running on Google Colab:
```python
!pip install catboost
```

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Download the dataset** from the [UCI repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) and place it in your working directory (or Google Drive if using Colab).

3. **Open the notebook** in Jupyter or Google Colab and update the file path in the data loading cell:
   ```python
   df = pd.read_excel('path/to/default of credit card clients.xlsx')
   ```

4. **Run all cells** from top to bottom.

---

## 📁 File Structure

```
├── Default_of_Credit_Card_Clients.ipynb   # Main notebook
└── README.md
```

---

## 📄 License

This project uses publicly available data from the UCI Machine Learning Repository. Please refer to the dataset's [original license](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) for usage terms.

---

## 🙏 Acknowledgements

- **Dataset:** I-Cheng Yeh, "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients," *Expert Systems with Applications*, 2009.
- UCI Machine Learning Repository for hosting the dataset.
