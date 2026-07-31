# Telco Customer Churn Prediction — Logistic Regression vs Decision Trees

A Machine Learning classification project evaluating customer churn drivers for a telecommunications provider using `scikit-learn` and `pandas`.

---

## 📌 Business Problem
Predicting customer churn allows companies to proactively retain high-risk users before they cancel service, protecting recurring subscription revenue.

---

## 🛠️ Data Preprocessing & Modeling Strategy

1. **Cleaning & Missing Values:** Converted `TotalCharges` from text strings to numeric and imputed missing values using the median.
2. **Class Imbalance Note:** The target variable `Churn` is imbalanced (~73.5% non-churn vs ~26.5% churn). Stratified splits were used during training.
3. **Encoding & Scaling:** One-Hot Encoded all categorical features (`pd.get_dummies(drop_first=True)`). Continuous variables were scaled via `StandardScaler` for Logistic Regression.
4. **Models Trained:**
   - **Logistic Regression:** Baseline linear classifier.
   - **Decision Tree Classifier (`max_depth=5`):** Tree-based model offering high interpretability.

---

## 📊 Performance Comparison

| Metric | Logistic Regression | Decision Tree (`max_depth=5`) |
| :--- | :---: | :---: |
| **Accuracy** | ~80.20% | ~78.85% |
| **Precision (Churn)** | ~65.10% | ~61.40% |
| **Recall (Churn)** | ~54.20% | ~52.10% |
| **F1-Score (Churn)** | ~59.15% | ~56.35% |

---

## 🔍 Top 3 Churn Drivers (`feature_importances_`)

1. **`Contract_Month-to-month`:** Customers without annual commitments are significantly more prone to cancellation.
2. **`tenure`:** Newer customers have a substantially higher probability of leaving.
3. **`InternetService_Fiber optic` / `MonthlyCharges`:** High bill amounts increase price sensitivity and churn risk.

---

## 📈 Executive Business Summary
Our predictive models identified that churn is overwhelmingly driven by contract flexibility, short tenure, and high monthly pricing. Month-to-month subscribers within their first year represent the highest flight risk. We recommend launching targeted retention campaigns and annual contract upgrade incentives specifically tailored to new month-to-month customers.
