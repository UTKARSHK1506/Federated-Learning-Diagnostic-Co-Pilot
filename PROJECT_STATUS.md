# Project Status & Technical Record

Technical record of verified values, hyperparameter configurations, and empirical baseline results for the **Federated-Learning Diagnostic Co-Pilot** repository.

---

## Technical Specifications & Verified Values

* **Dataset Size**: 70,000 raw patient records → **68,666** after blood-pressure outlier cleaning (`cardio_cleaned.csv`).
* **Non-IID Clinic Partitions**:
  * **Clinic A** (Older / Higher BP): 22,889 records (33.33%)
  * **Clinic B** (Younger / Lower BP): 22,889 records (33.33%)
  * **Clinic C** (Mixed Population): 22,888 records (33.33%)
  * *Duplication*: 0 cross-clinic duplicates (100% distinct patient IDs).
* **Model Features**: 11 clinical input features (`age`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`). `id` is excluded; `cardio` is the binary target.
* **Age Representation**: Converted from raw days to float years ($age / 365.25$).
* **Data Partitioning**: 80% Train ($N=18,311$), 10% Validation ($N=2,289$), 10% Test ($N=2,289$) per clinic using stratified splitting.
* **Scaling Strategy**: `StandardScaler` fitted **ONLY** on local clinic training data (`X_train`) to guarantee zero validation/test leakage.
* **Model Architecture**: Feed-Forward Neural Network (MLP): `Linear(11 → 64)` → `ReLU` → `Linear(64 → 32)` → `ReLU` → `Linear(32 → 1)` (returns raw logits).
* **Training Loss**: `BCEWithLogitsLoss`
* **Optimizer**: Adam ($\text{lr} = 0.001$)
* **Batch Size**: 64
* **Local Epochs per Round**: 1
* **Federated Rounds**: 5
* **Aggregation Strategy**: Weighted Federated Averaging (FedAvg) via Flower (`flwr`)
* **Random Seed**: 42

---

## Baseline Verification & Federated Results

### Local Training Baseline (Clinic A Only, 20 Epochs)

| Metric | Verified Value |
| :--- | :--- |
| **Test Loss** | `0.5584` |
| **Test Accuracy** | **72.04%** |
| **Precision** | `0.7491` |
| **Recall** | `0.7849` |
| **F1 Score** | `0.7666` |
| **Parameter Updates Verified** | **YES** |

### Final Global Federated Model Results (3 Clients, 5 FedAvg Rounds)

| Evaluation Test Set | Test Loss | Test Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Clinic A Test Set** | `0.5799` | **70.51%** | `0.7948` | `0.6684` | `0.7262` |
| **Clinic B Test Set** | `0.5626` | **71.82%** | `0.6421` | `0.6736` | `0.6575` |
| **Clinic C Test Set** | `0.5419` | **73.70%** | `0.7498` | `0.7076` | `0.7281` |

*Privacy Guarantee Verification*: Parameter arrays (weights & biases) were exchanged; 0 raw patient records were transmitted to the server.

---

## Milestone Completion Checklist

1. Dataset preprocessing — **DONE**
2. Non-IID clinic split — **DONE**
3. Preprocessing pipeline — **DONE**
4. FNN architecture — **DONE**
5. Local training — **DONE**
6. Federated Learning — **DONE**
