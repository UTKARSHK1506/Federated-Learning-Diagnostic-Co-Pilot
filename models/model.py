import torch
import torch.nn as nn


class DiagnosticModel(nn.Module):


    def __init__(self, input_size: int = 11, hidden_size1: int = 64, hidden_size2: int = 32):
        
        super(DiagnosticModel, self).__init__()

        # Layer 1: Input to first hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()

        # Layer 2: First hidden layer to second hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()

        # Layer 3: Second hidden layer to output logit
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        out = self.fc3(x)
        return out
