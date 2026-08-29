import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any

from models.model import DiagnosticModel
from preprocessing.preprocess import load_and_preprocess_clinic


def set_seed(seed: int = 42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_local_model(
    csv_path: str,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    random_seed: int = 42
) -> Tuple[DiagnosticModel, Dict[str, list], Dict[str, float], Any]:
    """
    Trains DiagnosticModel locally on a single clinic dataset.
    
    Args:
        csv_path (str): Path to the clinic CSV file (e.g., 'data/clinic_A.csv').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        learning_rate (float): Learning rate for Adam optimizer.
        random_seed (int): Seed for reproducible data splitting and weight initialization.
        
    Returns:
        Tuple containing:
            - trained_model (DiagnosticModel): PyTorch model after local training.
            - history (dict): Epoch-by-epoch lists of train_loss, val_loss, val_acc.
            - test_metrics (dict): Final evaluation metrics on local test set.
            - scaler (StandardScaler): Local scaler fitted ONLY on local training split.
    """
    set_seed(random_seed)

    # 1. Load and preprocess clinic data (80% train, 10% val, 10% test)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_clinic(
        csv_path=csv_path,
        val_ratio=0.10,
        test_ratio=0.10,
        random_seed=random_seed
    )

    clinic_name = os.path.basename(csv_path)
    print(f"\n==================================================")
    print(f"LOCAL TRAINING: {clinic_name}")
    print(f"==================================================")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    print(f"Hyperparameters:    Epochs={epochs}, Batch Size={batch_size}, LR={learning_rate}\n")

    # 2. Construct PyTorch DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Instantiate model, loss criterion, and optimizer
    model = DiagnosticModel(input_size=11, hidden_size1=64, hidden_size2=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Make a copy of initial model state dict for parameter update verification
    initial_state_dict = copy.deepcopy(model.state_dict())

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(X_batch)

        epoch_train_loss = running_train_loss / len(X_train)

        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * len(X_batch)
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(y_batch.cpu().numpy())

        epoch_val_loss = running_val_loss / len(X_val)
        epoch_val_acc = accuracy_score(val_targets_list, val_preds_list)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {epoch_val_acc*100:.2f}%")

    # 5. Final Test Set Evaluation
    model.eval()
    running_test_loss = 0.0
    test_preds_list = []
    test_targets_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_test_loss += loss.item() * len(X_batch)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            test_preds_list.extend(preds.cpu().numpy())
            test_targets_list.extend(y_batch.cpu().numpy())

    test_loss = running_test_loss / len(X_test)
    test_acc = accuracy_score(test_targets_list, test_preds_list)
    test_prec = precision_score(test_targets_list, test_preds_list, zero_division=0)
    test_rec = recall_score(test_targets_list, test_preds_list, zero_division=0)
    test_f1 = f1_score(test_targets_list, test_preds_list, zero_division=0)

    test_metrics = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1
    }

    # 6. Verify Model Parameters Changed
    final_state_dict = model.state_dict()
    parameters_updated = any(
        not torch.equal(initial_state_dict[key], final_state_dict[key])
        for key in initial_state_dict
    )

    print(f"\nFinal Test Results")
    print(f"------------------")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"Precision:      {test_prec:.4f}")
    print(f"Recall:         {test_rec:.4f}")
    print(f"F1 Score:       {test_f1:.4f}")
    print(f"\nModel parameters updated: {'YES' if parameters_updated else 'NO'}")
    print(f"\nModel architecture:")
    print(model)

    return model, history, test_metrics, scaler


if __name__ == '__main__':
    # Train locally on Clinic A
    clinic_a_path = os.path.join(project_root, 'data', 'clinic_A.csv')
    train_local_model(csv_path=clinic_a_path, epochs=20, batch_size=64, learning_rate=0.001, random_seed=42)
