import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.fc1 = torch.nn.Linear(d_model, 4 * d_model)

        self.fc2 = torch.nn.Linear(4 * d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
