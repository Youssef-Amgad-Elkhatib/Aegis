# 🫀 Aegis: Clinical-Grade Cardiovascular Risk Predictor

Aegis is a machine learning pipeline designed to predict cardiovascular disease risk based on routine medical vitals (Age, Height, Weight, Blood Pressure, Cholesterol, etc.). 

While many public notebooks claim 80-90% accuracy on this dataset by exploiting data leakage and biological typos, **Aegis focuses on mathematical integrity, clinical safety, and maximizing Recall (Sensitivity)**. The model achieves a robust **73.46% real-world accuracy**—the theoretical limit (Bayes Error Rate) of this dataset given the vast amount of identical overlapping patient profiles.

---

## 🔬 Special Focus: Advanced Data Preprocessing & Cleaning

The core differentiator of this project is its meticulous approach to data cleaning and feature engineering. Medical datasets are notoriously noisy, and failing to handle this noise leads to catastrophic data leakage.

### 1. Eliminating "Biological Typos"
The raw dataset contained severe data entry errors that tree-based algorithms easily exploit to artificially inflate accuracy (e.g., associating extreme typos solely with sick patients).
* **Blood Pressure Anomalies:** Removed impossible systolic (`ap_hi` > 300, max was 16,020) and diastolic (`ap_lo` > 200, max was 11,000) readings, as well as negative pressures.
* **Physical Constraints:** Removed physiologically impossible heights (e.g., 55 cm adults) and weights to prevent the model from learning medical anomalies.

### 2. Solving the "Clone" Problem & Data Leakage
The dataset contained **8,719 overlapping profiles**—patients with the exact same vitals but different cardiovascular outcomes (The Contradiction Wall).
* **ID Dropping:** The unique `id` column was dropped *before* checking for duplicates. (Checking with IDs present falsely returns 0 duplicates).
* **Leakage Prevention:** By identifying and handling these clones prior to the `train_test_split`, preventing identical patient vectors from appearing in both the Training and Testing sets, ensuring the model's test score represents true generalization, not memorization.

### 3. Clinical Feature Engineering
To give the algorithms a "shortcut" to medical domain knowledge, several composite clinical features were engineered from the raw vitals:
* **Pulse Pressure:** `ap_hi - ap_lo` (A strong indicator of arterial stiffness).
* **Mean Arterial Pressure (MAP):** `(2/3 * ap_lo) + (1/3 * Pulse Pressure)` (Average pressure in a patient's arteries during one cardiac cycle).

---

## 🧠 Model Training & Validation

To completely prevent overfitting, the data was split using a strict 3-way methodology: **60% Training | 20% Validation | 20% Testing**.

### Custom Hyperparameter Tuning Engine
Instead of relying on standard GridSearch (which only optimizes for the highest raw score), a custom randomized search function was built to track the **absolute difference between Training and Validation metrics**. 
* **The Threshold:** Models were rejected if the Train/Val gap exceeded `2.5%`.
* **The Result:** The tuning engine heavily penalized deep, over-fitted trees, settling on a generalized architecture.

### The Winning Architecture: Gradient Boosting
* `n_estimators`: 300
* `max_depth`: 5 (Shallow trees to prevent memorization)
* `max_features`: 'sqrt' (Forces generalized learning across different feature sets)
* `subsample`: 0.9 (Uses 90% of data per tree, increasing robustness against noise)
* `learning_rate`: 0.05

---

## 📊 Final Evaluation & Clinical Strategy

### Real-World Performance (Unseen Test Vault: 13,646 patients)
* **Accuracy:** 73.46%
* **Precision:** 75% (When it predicts sick, it is correct 75% of the time)
* **Recall:** 73% (It can detect 73% of real sick people)

### Prioritizing Recall (Sensitivity)
In the medical domain, False Negatives (sending a sick patient home) are vastly more dangerous than False Positives (sending a healthy patient for a blood test). 

---
*Built with Scikit-Learn, Pandas, NumPy, and Seaborn.*
