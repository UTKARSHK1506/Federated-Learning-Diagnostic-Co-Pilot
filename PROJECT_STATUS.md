# Project Status & Technical Record

Technical record of verified values, hyperparameter configurations, baseline results, and dual-mode inference architecture for the **Federated-Learning Diagnostic Co-Pilot** repository.

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

## Dual Inference Architecture (Online & Offline Modes)

The application supports two unified inference modes wrapped behind the `InferenceRepository` abstraction, returning identical output data structures (`predicted_class`, `probability`):

1. **ONLINE MODE**:
   * **Path**: `Android Application` → `FastAPI REST API (/predict)` → `PyTorch global_model.pt` → `Inference Response`
   * **Behavior**: Communicates over HTTP with the central FastAPI backend. Used when connected to the clinical network.

2. **OFFLINE MODE**:
   * **Path**: `Android Application` → `Local Preprocessing (scaler_params.json)` → `Bundled ONNX / Local Engine (global_model.onnx)` → `Inference Response`
   * **Behavior**: Performs local standardization of continuous features ($x_{\text{std}} = (x - \mu)/\sigma$) and executes local neural network matrix forward passes on-device with zero network latency, working seamlessly when Wi-Fi/Server is offline.

3. **Model Conversion Format**:
   * Converted from `global_model.pt` to **ONNX format (`global_model.onnx`)** and exported JSON matrix weights (`global_model_weights.json`).
   * Scaler parameters exported to `scaler_params.json` (`mean_` and `scale_` arrays for `age`, `height`, `weight`, `ap_hi`, `ap_lo`).

4. **Equivalence Verification**:
   * **Online Mode (FastAPI)**: Predicted Class `1`, Probability `0.7388` (`73.88%`)
   * **Offline Mode (Local Engine)**: Predicted Class `1`, Probability `0.7388` (`73.88%`)
   * **Absolute Difference**: `0.000000` (**100% Exact Equivalence Match**)

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

### Final Global Federated Model Results (3 Clients, 5 FedAvg Rounds)

| Evaluation Test Set | Test Loss | Test Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Clinic A Test Set** | `0.5791` | **70.07%** | `0.7957` | `0.6572` | `0.7198` |
| **Clinic B Test Set** | `0.5606` | **72.04%** | `0.6473` | `0.6670` | `0.6570` |
| **Clinic C Test Set** | `0.5401` | **73.70%** | `0.7545` | `0.6989` | `0.7256` |

---

## Milestone Completion Checklist

1. Dataset preprocessing — **DONE**
2. Non-IID clinic split — **DONE**
3. Preprocessing pipeline — **DONE**
4. FNN architecture — **DONE**
5. Local training — **DONE**
6. Federated Learning — **DONE**
7. FastAPI REST Backend — **DONE**
8. Dual Online/Offline Inference — **DONE**
