
# 🧠 Machine Learning Approaches to Retail Credit Risk Prediction

## 🎓 Graduation Thesis – Industrial Engineering

**Title:** Machine Learning Approaches to Retail Credit Risk Prediction: Comparative Analysis of XGBoost, Random Forest, and Logistic Regression

**Authors:** [Hande Kubilay](https://github.com/handekubilay), Utku Pehlivan

**University:** Yildiz Technical University

**Department:** Industrial Engineering

**Graduation Year:** 2025

---

## 📌 Project Description

This thesis project investigates the application of supervised machine learning algorithms in the context of **retail credit risk prediction**. Specifically, it provides a **comparative performance analysis** of three prominent models:

* Logistic Regression (baseline model)
* Random Forest (ensemble method)
* XGBoost (gradient boosting technique)

By developing and evaluating these models on a real-world credit dataset, the project aims to assess their ability to **predict default risk** and support decision-making in financial institutions.

---

## 🎯 Objectives

* To apply and optimize multiple machine learning models for classifying borrowers as **creditworthy** or **high-risk**.
* To handle class imbalance effectively using **Synthetic Minority Over-sampling Technique (SMOTE)**.
* To evaluate model performance using appropriate metrics for imbalanced classification problems.
* To analyze and compare the strengths and limitations of **traditional (Logistic Regression)** and **ensemble methods (Random Forest, XGBoost)**.

---

## 🧪 Methodology

### 1. Data Preprocessing & Feature Engineering

* Missing value handling
* Categorical variable encoding (e.g., one-hot encoding, label encoding)
* Feature scaling and transformation

### 2. Handling Class Imbalance

* Use of **SMOTE** (Synthetic Minority Over-sampling Technique) to balance target classes and mitigate model bias.

### 3. Model Building & Optimization

* Implementation of Logistic Regression, Random Forest, and XGBoost
* **Cross-validation** to ensure robust model performance
* **Grid Search** and **Randomized Search** for hyperparameter tuning

### 4. Model Evaluation

* Confusion Matrix Analysis
* Key metrics: Accuracy, Precision, Recall, F1-score, Log Loss
* **Trade-off analysis** between Type I (False Positive) and Type II (False Negative) errors
* ROC and Precision-Recall Curves

---

## 📊 Performance Metrics

| Metric        | Description                                                                  |
| ------------- | ---------------------------------------------------------------------------- |
| **Accuracy**  | Overall correctness of the model                                             |
| **Precision** | Correctly predicted positives among all predicted positives (↓ Type I error) |
| **Recall**    | Correctly predicted positives among all actual positives (↓ Type II error)   |
| **F1-score**  | Harmonic mean of Precision and Recall                                        |
| **Log Loss**  | Measures uncertainty of classification probabilities                         |

---

## 🧰 Tools & Technologies

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * `scikit-learn` – ML model implementation
  * `xgboost` – Gradient boosting classifier
  * `pandas`, `numpy` – Data manipulation
  * `matplotlib`, `seaborn` – Visualization
  * `imbalanced-learn` – SMOTE and resampling tools

---

## 📁 Repository Structure

```bash
credit-risk-ml/
│
├── credit_risk_ml.ipynb         # Main notebook with analysis and results
├── credit_risk_ml_outputs/      # Figures and model output images
├── README.md                    # Project overview
```

---

## 🔍 Key Findings

* **XGBoost** achieved the highest overall performance across most metrics due to its ability to handle complex patterns and class imbalance.
* **Logistic Regression**, while interpretable, struggled with recall, making it less reliable for minimizing false negatives.
* **Random Forest** offered a balance between interpretability and performance but required tuning to avoid overfitting.

> The results indicate that ensemble methods, particularly gradient boosting, are well-suited for credit scoring applications when properly tuned.

---

## 📬 Contact

For any questions, collaboration opportunities, or feedback, feel free to reach out:

**Hande Kubilay**
📧 [handekubilay35@gmail.com](mailto:handekubilay35@gmail.com)
📍 Istanbul, Turkiye

---

## ✅ Future Work

* Incorporate more advanced sampling techniques (e.g., ADASYN, Tomek Links)
* Extend analysis using real bank credit portfolios (with anonymized data)
* Investigate explainability techniques (e.g., SHAP values, LIME) for regulatory compliance

---
