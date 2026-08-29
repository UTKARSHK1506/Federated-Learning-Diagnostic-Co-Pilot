import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, List

from models.model import DiagnosticModel
from preprocessing.preprocess import load_and_preprocess_clinic


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Extracts model weights and biases as a list of NumPy arrays for Flower parameter exchange.
    
    Exchanged Tensors:
      - fc1.weight, fc1.bias
      - fc2.weight, fc2.bias
      - fc3.weight, fc3.bias
    """
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    """
    Loads a list of NumPy weight/bias arrays received from Flower into the PyTorch model's state_dict.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


class ClinicFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient representing an isolated clinic node (Clinic A, Clinic B, or Clinic C).
    
    Data Privacy & Isolation:
    - Raw patient records (X_train, y_train, CSV contents) remain strictly local within this client node.
    - Only model parameter arrays (weights and biases) and scalar metrics are exchanged with the server.
    """

    def __init__(
        self,
        csv_path: str,
        local_epochs: int = 1,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        random_seed: int = 42
    ):
        self.csv_path = csv_path
        self.clinic_id = os.path.basename(csv_path).replace('.csv', '').replace('clinic_', 'Clinic ')
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 1. Load and preprocess ONLY local clinic CSV data (80% train, 10% val, 10% test)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.scaler = load_and_preprocess_clinic(
            csv_path=self.csv_path,
            val_ratio=0.10,
            test_ratio=0.10,
            random_seed=random_seed
        )

        # 2. Instantiate local PyTorch DiagnosticModel (11 -> 64 -> 32 -> 1)
        self.model = DiagnosticModel(input_size=11, hidden_size1=64, hidden_size2=32)
        self.criterion = nn.BCEWithLogitsLoss()

    def get_parameters(self, config: Dict[str, str] = None) -> List[np.ndarray]:
        """
        Returns current local model weights and biases to Flower server.
        """
        return get_model_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str] = None) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        Performs local training on local clinic data using global parameters sent by the server.
        """
        set_model_parameters(self.model, parameters)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        running_loss = 0.0
        for _ in range(self.local_epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(X_batch)

        avg_train_loss = running_loss / (len(self.X_train) * self.local_epochs)
        
        print(f"[{self.clinic_id}] Local training complete ({self.local_epochs} epoch) | Local Train Loss: {avg_train_loss:.4f}")

        return get_model_parameters(self.model), len(self.X_train), {"train_loss": float(avg_train_loss)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str] = None) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluates received model parameters on the clinic's local validation dataset.
        """
        set_model_parameters(self.model, parameters)

        val_dataset = TensorDataset(self.X_val, self.y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        running_val_loss = 0.0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                running_val_loss += loss.item() * len(X_batch)

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                preds_list.extend(preds.cpu().numpy())
                targets_list.extend(y_batch.cpu().numpy())

        val_loss = running_val_loss / len(self.X_val)
        val_acc = accuracy_score(targets_list, preds_list)
        val_f1 = f1_score(targets_list, preds_list, zero_division=0)

        return float(val_loss), len(self.X_val), {"val_loss": float(val_loss), "accuracy": float(val_acc), "f1": float(val_f1)}

    def evaluate_test_set(self) -> Dict[str, float]:
        """
        Evaluates the model on the local test set after federated learning completion.
        """
        test_dataset = TensorDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        running_test_loss = 0.0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                running_test_loss += loss.item() * len(X_batch)

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                preds_list.extend(preds.cpu().numpy())
                targets_list.extend(y_batch.cpu().numpy())

        test_loss = running_test_loss / len(self.X_test)
        acc = accuracy_score(targets_list, preds_list)
        prec = precision_score(targets_list, preds_list, zero_division=0)
        rec = recall_score(targets_list, preds_list, zero_division=0)
        f1 = f1_score(targets_list, preds_list, zero_division=0)

        return {
            "test_loss": float(test_loss),
            "test_accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
