# CardioSense — Model Evaluation Metrics

## 1. Evaluation Overview

This document presents a complete, fresh empirical evaluation of the **CardioSense ML and Federated Learning implementation**. All metrics reported herein were obtained from direct execution of training and evaluation scripts on held-out clinic test sets using PyTorch (`2.13.0+cpu`), Scikit-Learn (`1.7.2`), and Flower (`1.30.0`).

No model weights, scaling parameters, neural network architectures, or app inference logic were modified for this evaluation.

---

## 2. Dataset

* **Source Dataset**: Kaggle Cardiovascular Disease Dataset (70,000 raw patient records).
* **Cleaned Dataset Size**: **68,666 patient records** (after removing invalid blood pressure outliers, e.g. $\text{ap\_hi} \le 70$ or $\text{ap\_hi} \ge 240$, $\text{ap\_lo} \le 40$ or $\text{ap\_lo} \ge 140$).
* **Features (11 Input Clinical Variables)**:
  1. `age`: Patient age in float years (converted from raw days / 365.25).
  2. `gender`: Biological sex (`1` = Female, `2` = Male).
  3. `height`: Stature in centimeters ($\text{cm}$).
  4. `weight`: Body mass in kilograms ($\text{kg}$).
  5. `ap_hi`: Systolic blood pressure ($\text{mmHg}$).
  6. `ap_lo`: Diastolic blood pressure ($\text{mmHg}$).
  7. `cholesterol`: Serum cholesterol level (`1` = Normal, `2` = Above Normal, `3` = Well Above Normal).
  8. `gluc`: Fasting blood glucose level (`1` = Normal, `2` = Above Normal, `3` = Well Above Normal).
  9. `smoke`: Smoking status (`0` = Non-smoker, `1` = Active smoker).
  10. `alco`: Alcohol consumption (`0` = Non-drinker, `1` = Alcohol consumer).
  11. `active`: Physical activity (`0` = Inactive / Sedentary, `1` = Physically active).
* **Target Variable**: `cardio` (Binary classification: `0` = Absence of Cardiovascular Disease, `1` = Presence of Cardiovascular Disease).

---

## 3. Data Partitioning

The cleaned dataset ($N=68,666$) was partitioned across **3 simulated hospital/clinic nodes** to reflect realistic **Non-IID (heterogeneous)** clinical demographics:

* **Clinic A (Older / High BP Cohort)**: $N=22,889$ records (33.33%). Partitioned using a composite score biased toward older patients with elevated blood pressure.
* **Clinic B (Younger / Lower BP Cohort)**: $N=22,889$ records (33.33%). Partitioned using a composite score biased toward younger patients with lower blood pressure.
* **Clinic C (Mixed Population Cohort)**: $N=22,888$ records (33.33%). Partitioned from the remaining balanced population.
* **Data Isolation**: 0 duplicate patient IDs across clinics (100% strict patient record privacy across all 3 nodes).

---

## 4. Preprocessing

* **Data Splitting**: Each clinic independently performs a **stratified 80% / 10% / 10% split**:
  * **Local Training Set (`X_train`)**: 80% ($N=18,311$)
  * **Local Validation Set (`X_val`)**: 10% ($N=2,289$)
  * **Local Test Set (`X_test`)**: 10% ($N=2,289$)
* **StandardScaler Usage**: Continuous features (`age`, `height`, `weight`, `ap_hi`, `ap_lo`) are z-score standardized using `sklearn.preprocessing.StandardScaler`.
* **Zero Data Leakage**: The `StandardScaler` is fitted **exclusively on `X_train`** for each clinic. `X_val`, `X_test`, and cross-clinic test cohorts are transformed using the pre-fitted training scaler without refitting.

---

## 5. Model Architecture

The deep learning model is a Feed-Forward Neural Network (Multi-Layer Perceptron):

```text
Input (11 features) ──► Linear(11, 64) ──► ReLU ──► Linear(64, 32) ──► ReLU ──► Linear(32, 1) ──► Raw Logit
```

* **Input Layer**: 11 continuous/categorical clinical features.
* **Hidden Layer 1**: Fully Connected 64 units + ReLU activation.
* **Hidden Layer 2**: Fully Connected 32 units + ReLU activation.
* **Output Layer**: Fully Connected 1 unit producing a continuous unbounded logit output $z$.
* **Inference Probability**: Computed on-demand via Sigmoid: $P(y=1) = \frac{1}{1 + e^{-z}}$.

---

## 6. Training Configuration

* **Loss Function**: `torch.nn.BCEWithLogitsLoss()` (combines a Sigmoid layer and Binary Cross-Entropy in one single numerically stable class).
* **Optimizer**: Adam ($\text{lr} = 0.001$).
* **Batch Size**: 64 samples.
* **Local Baseline Training**: 20 local epochs per clinic.
* **Random Seed**: `42` (for 100% reproducible data splits, DataLoader shuffling, and weight initialization).

---

## 7. Local Model Results

Each clinic model was trained locally on its own local training split ($N=18,311$) for 20 epochs and evaluated on its local test set ($N=2,289$):

| Test Cohort | Model | Test Loss | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Test Samples |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Clinic A Test Set** | Clinic A Local Baseline | `0.5584` | **72.04%** | `0.7491` | **0.7849** | **0.7666** | `0.7739` | 2,289 |
| **Clinic B Test Set** | Clinic B Local Baseline | `0.5442` | **73.09%** | `0.7056` | `0.5658` | `0.6280` | `0.7780` | 2,289 |
| **Clinic C Test Set** | Clinic C Local Baseline | `0.5404` | **73.04%** | `0.7426` | `0.7015` | `0.7214` | **0.8011** | 2,289 |

---

## 8. Federated Learning Configuration

* **Federated Protocol**: Flower (`flwr`) NumPyClient architecture.
* **Aggregation Algorithm**: Weighted Federated Averaging (`FedAvg`):
  $$W_{\text{global}} = \sum_{i=1}^{K} \frac{N_i}{N_{\text{total}}} W_i$$
* **Number of Clients**: 3 hospital nodes (Clinic A, Clinic B, Clinic C).
* **Federated Communication Rounds**: 5 rounds.
* **Local Epochs per Round**: 1 local epoch per round per client.
* **Transmitted Content**: Only model parameter arrays (weights and biases). **0 raw patient records transferred**.

---

## 9. Federated Global Model Results

After 5 rounds of federated training, the final global model was evaluated across all three distinct hospital test cohorts:

| Test Cohort | Model | Test Loss | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Test Samples |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Clinic A Test Set** | Federated Global Model | `0.5781` | **70.77%** | **0.7975** | `0.6706` | `0.7286` | `0.7763` | 2,289 |
| **Clinic B Test Set** | Federated Global Model | `0.5615` | **71.60%** | `0.6371` | `0.6801` | `0.6579` | `0.7806` | 2,289 |
| **Clinic C Test Set** | Federated Global Model | `0.5396` | **74.01%** | `0.7552` | **0.7068** | **0.7302** | **0.8036** | 2,289 |
| **Combined Test Cohorts (Micro-Avg)** | Federated Global Model | `0.5597` | **72.13%** | `0.7337` | `0.6853` | `0.7087` | `0.7841` | 6,867 |
| **Combined Test Cohorts (Macro-Avg)** | Federated Global Model | `0.5597` | **72.13%** | `0.7299` | `0.6858` | `0.7056` | `0.7868` | 6,867 |

---

## 10. Local vs Federated Comparison

| Test Cohort | Model Type | Test Loss | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Clinic A** (Older/High BP) | Local Baseline | `0.5584` | **72.04%** | `0.7491` | **0.7849** | **0.7666** | `0.7739` |
| **Clinic A** (Older/High BP) | Federated Global | `0.5781` | 70.77% | **0.7975** | `0.6706` | `0.7286` | `0.7763` |
| **Clinic B** (Younger/Low BP) | Local Baseline | `0.5442` | **73.09%** | **0.7056** | `0.5658` | `0.6280` | `0.7780` |
| **Clinic B** (Younger/Low BP) | Federated Global | `0.5615` | 71.60% | `0.6371` | **0.6801** | **0.6579** | **0.7806** |
| **Clinic C** (Mixed Population)| Local Baseline | `0.5404` | 73.04% | `0.7426` | `0.7015` | `0.7214` | `0.8011` |
| **Clinic C** (Mixed Population)| Federated Global | `0.5396` | **74.01%** | **0.7552** | **0.7068** | **0.7302** | **0.8036** |

---

## 11. Confusion Matrices

### Local Models (20 Epochs)

* **Clinic A Local Model on Clinic A Test Set**:
  * True Negatives (TN): **598**
  * False Positives (FP): **352**
  * False Negatives (FN): **288**
  * True Positives (TP): **1,051**
* **Clinic B Local Model on Clinic B Test Set**:
  * True Negatives (TN): **1,153**
  * False Positives (FP): **217**
  * False Negatives (FN): **399**
  * True Positives (TP): **520**
* **Clinic C Local Model on Clinic C Test Set**:
  * True Negatives (TN): **873**
  * False Positives (FP): **277**
  * False Negatives (FN): **340**
  * True Positives (TP): **799**

### Federated Global Model (5 Rounds)

* **Global Model on Clinic A Test Set**:
  * True Negatives (TN): **722**
  * False Positives (FP): **228**
  * False Negatives (FN): **441**
  * True Positives (TP): **898**
* **Global Model on Clinic B Test Set**:
  * True Negatives (TN): **1,014**
  * False Positives (FP): **356**
  * False Negatives (FN): **294**
  * True Positives (TP): **625**
* **Global Model on Clinic C Test Set**:
  * True Negatives (TN): **889**
  * False Positives (FP): **261**
  * False Negatives (FN): **334**
  * True Positives (TP): **805**

---

## 12. Class Distribution

| Clinic Test Cohort | Total Test Samples | Class 0 (No Disease) | Class 0 % | Class 1 (Disease Present) | Class 1 % |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Clinic A Test Set** | 2,289 | 950 | 41.50% | 1,339 | 58.50% |
| **Clinic B Test Set** | 2,289 | 1,370 | 59.85% | 919 | 40.15% |
| **Clinic C Test Set** | 2,289 | 1,150 | 50.24% | 1,139 | 49.76% |

---

## 13. Cross-Clinic Generalization

1. **Peak Generalization Performance**: The federated global model achieved its highest overall accuracy (**74.01%**), highest F1-score (**0.7302**), and best ROC-AUC (**0.8036**) on the mixed-population **Clinic C** test set, surpassing Clinic C's local baseline model.
2. **Improved Recall on Skewed Demographics**: On Clinic B (younger cohort), local training produced severe false negatives (Recall = 56.58%). Federated aggregation improved Clinic B's recall significantly to **68.01%** (a **+11.43% improvement**), enabling better detection of cardiovascular disease in younger patients.
3. **Precision Boost on High-Risk Demographics**: On Clinic A (older/high BP cohort), federated aggregation increased precision from **0.7491 to 0.7975** (a **+4.84% improvement**), reducing false positive diagnoses.

---

## 14. Online vs Offline Inference Verification

> [!NOTE]
> This section measures **Inference-Equivalence Verification**, NOT dataset model evaluation accuracy.

CardioSense supports both centralized cloud inference (FastAPI) and on-device offline inference (Android Kotlin):

* **Sample Test Patient**: `Age: 55.5`, `Gender: 1 (Female)`, `Height: 165.0 cm`, `Weight: 70.0 kg`, `Systolic BP: 140 mmHg`, `Diastolic BP: 90 mmHg`, `Cholesterol: Level 2`, `Glucose: Level 1`, `Smoke: 0`, `Alcohol: 0`, `Active: 1`.
* **Online Prediction (FastAPI PyTorch Server)**: Class `1` (Elevated Risk), Probability = **0.7388** (`73.88%`).
* **Offline Prediction (Android Kotlin Engine)**: Class `1` (Elevated Risk), Probability = **0.7388** (`73.88%`).
* **Absolute Numerical Difference**: **`0.000000`** (**100% Exact Equivalence**).

