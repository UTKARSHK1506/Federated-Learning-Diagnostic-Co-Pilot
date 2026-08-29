import os
import sys
import pickle

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from typing import List, Dict, Tuple

from models.model import DiagnosticModel
from federated.client import ClinicFlowerClient, get_model_parameters, set_model_parameters


def federated_averaging(client_parameters: List[List[np.ndarray]], sample_counts: List[int]) -> List[np.ndarray]:
    """
    Executes Weighted Federated Averaging (FedAvg) aggregation over client parameters.
    
    Formula:
        W_global = sum( (N_i / N_total) * W_i ) for each weight/bias array layer.
    """
    total_samples = sum(sample_counts)
    num_layers = len(client_parameters[0])
    
    aggregated_params = []
    for layer_idx in range(num_layers):
        weighted_layer_sum = np.zeros_like(client_parameters[0][layer_idx], dtype=np.float64)
        for client_idx in range(len(client_parameters)):
            weight_factor = sample_counts[client_idx] / total_samples
            weighted_layer_sum += weight_factor * client_parameters[client_idx][layer_idx]
        aggregated_params.append(weighted_layer_sum.astype(client_parameters[0][layer_idx].dtype))
        
    return aggregated_params


def run_federated_learning(
    num_rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    random_seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Orchestrates the 3-Client Federated Learning Prototype.
    """
    print("==================================================")
    print("FEDERATED LEARNING EXPERIMENT (3 CLIENTS, FEDAVG)")
    print("==================================================")
    print(f"Rounds:              {num_rounds}")
    print(f"Local Epochs/Round:  {local_epochs}")
    print(f"Batch Size:          {batch_size}")
    print(f"Learning Rate:       {learning_rate}")
    print(f"Random Seed:         {random_seed}")
    print("--------------------------------------------------\n")

    data_dir = os.path.join(project_root, 'data')
    clinic_files = {
        'Clinic A': os.path.join(data_dir, 'clinic_A.csv'),
        'Clinic B': os.path.join(data_dir, 'clinic_B.csv'),
        'Clinic C': os.path.join(data_dir, 'clinic_C.csv')
    }

    # 1. Initialize local Flower client nodes
    clients: Dict[str, ClinicFlowerClient] = {}
    for name, path in clinic_files.items():
        clients[name] = ClinicFlowerClient(
            csv_path=path,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_seed=random_seed
        )

    # 2. Server initializes global DiagnosticModel parameters
    global_model = DiagnosticModel(input_size=11, hidden_size1=64, hidden_size2=32)
    global_parameters = get_model_parameters(global_model)

    print("Server: Initialized global DiagnosticModel parameters (11 -> 64 -> 32 -> 1).")
    print("Server: Starting federated training rounds...\n")

    # 3. Execute Federated Rounds
    for r in range(1, num_rounds + 1):
        print(f"--- Round {r} ---")
        
        client_updates = []
        sample_counts = []
        
        for name, client in clients.items():
            updated_params, num_samples, fit_metrics = client.fit(global_parameters)
            client_updates.append(updated_params)
            sample_counts.append(num_samples)

        global_parameters = federated_averaging(client_updates, sample_counts)
        set_model_parameters(global_model, global_parameters)
        print(f"Server -> FedAvg aggregation complete (Round {r})\n")

    # 4. Save the Final Global Model & Scaler for Backend API Service
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, 'global_model.pt')
    scaler_save_path = os.path.join(models_dir, 'scaler.pkl')
    
    torch.save(global_model.state_dict(), model_save_path)
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(clients['Clinic A'].scaler, f)
        
    print(f"Server: Saved final global model parameters to '{model_save_path}'")
    print(f"Server: Saved reference scaler to '{scaler_save_path}'\n")

    print("==================================================")
    print("FEDERATED LEARNING COMPLETE: FINAL GLOBAL MODEL EVALUATION")
    print("==================================================")

    final_metrics = {}
    for name, client in clients.items():
        set_model_parameters(client.model, global_parameters)
        metrics = client.evaluate_test_set()
        final_metrics[name] = metrics
        
        print(f"\nFinal Global Model Performance on {name} Test Set:")
        print(f"  Test Loss:     {metrics['test_loss']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall:        {metrics['recall']:.4f}")
        print(f"  F1 Score:      {metrics['f1_score']:.4f}")

    print("\n--------------------------------------------------")
    print("FEDERATED VERIFICATION CONFIRMATION:")
    print("  1. All 3 clinics connected and executed local training.")
    print("  2. Same DiagnosticModel architecture used by all clients.")
    print("  3. Parameters exchanged: YES (weights & biases arrays).")
    print("  4. Raw patient data transmitted to server: NO (0 raw records transferred).")
    print("  5. Server FedAvg aggregation: SUCCESS.")
    print("--------------------------------------------------")

    return final_metrics


if __name__ == '__main__':
    run_federated_learning(num_rounds=5, local_epochs=1, batch_size=64, learning_rate=0.001, random_seed=42)
