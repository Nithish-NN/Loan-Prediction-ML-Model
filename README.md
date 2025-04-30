
# Feature Forge: Enhancing & Evaluating ML Models – Loan Prediction Project

This project demonstrates a full machine learning pipeline on the **Loan Prediction Dataset** from Analytics Vidhya. It covers comprehensive feature engineering, model tuning, evaluation, and deployment readiness.

## 🧠 Objective
Predict whether a loan will be approved (`Loan_Status`) based on applicant details. The goal is to maximize the F1-score while ensuring the model is robust and production-ready.

---

## 📁 Dataset Description

The dataset contains information on loan applicants, such as gender, income, credit history, and property area.

### 🔹 Train Dataset (Used for model training and validation)
- **Features:** `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`
- **Target:** `Loan_Status`

### 🔹 Test Dataset (Used for final prediction)
- Same features as the train dataset (excluding `Loan_Status`)

---

## 🔄 ML Pipeline Steps

### 1. 📊 Data Exploration
- Visualized target class distribution and key feature distributions.
- Analyzed relationships between categorical features and loan status.

### 2. 🧱 Feature Engineering
- Imputation for missing values.
- Label encoding and one-hot encoding.
- Created new features like:
  - `Total_Income`
  - `Loan_Income_Ratio`
  - `EMI`
  - Log transformations for skewed variables.

### 3. 🎯 Feature Selection
- Used Random Forest to assess feature importances.
- Selected top 10 features for model training.

### 4. 🏗️ Model Architecture & Tuning
- Compared **Random Forest** and **XGBoost** models.
- Performed hyperparameter tuning using `GridSearchCV`.

### 5. 🔁 Cross-Validation & Regularization
- 5-fold cross-validation to validate generalization.
- Applied max-depth and L2 regularization.

### 6. 📈 Performance Evaluation
- Evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Displayed Confusion Matrix and ROC Curve.

### 7. ⚠️ Stress Testing
- Simulated extreme test cases (e.g., high loan amount) to validate robustness.

### 8. 🚀 Deployment Pipeline
- Created a deployment-ready pipeline using `Pipeline` and `joblib`.
- Flask API structure provided for production deployment.

---

## 🛠️ Tech Stack

- **Languages:** Python 3.x
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib, xgboost
- **Tools:** Jupyter Notebook, Flask (optional), joblib

---
