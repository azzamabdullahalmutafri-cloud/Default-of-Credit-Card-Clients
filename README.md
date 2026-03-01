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
- Dataset shape, data types, and missing value checks
- Descriptive statistics
- Distribution plots and correlation analysis
- Class imbalance inspection

### 2. Preprocessing & Feature Engineering
- Handling class imbalance via `scale_pos_weight`
- Train/test splitting with stratification
- Pipelines with `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning

### 3. Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| K-Nearest Neighbors (KNN) | Distance-based classifier |
| MLP (Neural Network) | Multilayer Perceptron via scikit-learn |
| XGBoost | Gradient boosting with tuned hyperparameters |
| CatBoost | Gradient boosting optimized for categorical features |

### 4. Evaluation Strategy
- **10-Fold Stratified Cross-Validation** (out-of-fold predictions)
- Metrics: Accuracy, ROC-AUC, Precision, Recall, F1-Score
- Custom probability threshold (0.3) to improve recall on the minority class

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| CatBoost | 76.5% | 0.787 |
| XGBoost | 76.2% | 0.786 |

> CatBoost achieved the best overall performance with a ROC-AUC of **0.787** and an F1-score of **0.54** on the default class.

---

## 🛠️ Requirements

```bash
pip install catboost xgboost scikit-learn pandas numpy matplotlib seaborn
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
