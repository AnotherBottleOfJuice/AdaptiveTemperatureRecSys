import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from torch import nn
from dataset import TrainingBatch

if TYPE_CHECKING:
    from .graph import Graph


class Tau(nn.Module, ABC):
    def __init__(self,
                 initial_tau: float = 0.3,
                 tau_min: float = 0.055,
                 tau_max: float = 0.06,
                 num_epochs: int = 5,
                 num_tokens_per_epoch: int = 4_019_032) -> None:
        super().__init__()
        self.initial_tau = initial_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.num_epochs = num_epochs
        self.num_tokens_per_epoch = num_tokens_per_epoch

    @abstractmethod
    def __call__(self, net: 'Graph',
                 pos_logits: torch.Tensor,
                 neg_logits: torch.Tensor,
                 batch: TrainingBatch, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def experiment_name(self) -> str:
        pass

class ConstantTau(Tau):
    def __call__(self, net: 'Graph',
                 pos_logits: torch.Tensor,
                 neg_logits: torch.Tensor,
                 batch: TrainingBatch, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(self.initial_tau, device=pos_logits.device, dtype=torch.float32)

    def experiment_name(self) -> str:
        return f"Constant[value={self.initial_tau:.5g}]"

class ParameterTau(Tau):
    def __init__(self,
                 initial_tau: float = 0.045,
                 tau_min: float = 0.03,
                 tau_max: float = 0.06,
                 num_epochs: int = 5,
                 num_tokens_per_epoch: int = 4_019_032) -> None:
        super().__init__(initial_tau=initial_tau,
                         tau_min=tau_min,
                         tau_max=tau_max,
                         num_epochs=num_epochs,
                         num_tokens_per_epoch=num_tokens_per_epoch)
        self.initial_tau = float(initial_tau)
        self.tau = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def __call__(self, net: 'Graph',
                 pos_logits: torch.Tensor,
                 neg_logits: torch.Tensor,
                 batch: TrainingBatch, *args, **kwargs) -> torch.Tensor:
        tau_normalized = torch.sigmoid(self.tau)
        tau = self.tau_min + tau_normalized * (self.tau_max - self.tau_min)
        return tau.to(pos_logits.device)

    def experiment_name(self) -> str:
        return f"Param[init={self.initial_tau:.5g}]"

class LinearTau(Tau):
    def __call__(self, net: 'Graph',
                 pos_logits: torch.Tensor,
                 neg_logits: torch.Tensor,
                 batch: TrainingBatch, *args, **kwargs) -> torch.Tensor:
        tokens_passed = net.tokens_passed.to(device=pos_logits.device, dtype=torch.float32)
        phase = tokens_passed / (float(self.num_tokens_per_epoch) * float(self.num_epochs))
        return self.tau_max + phase * (self.tau_min - self.tau_max)

    def experiment_name(self) -> str:
        return (
            f"Linear[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g},"
            f"epochs={int(self.num_epochs)}]"
        )

class CosTau(Tau):
    def __call__(self, net: 'Graph',
                 pos_logits: torch.Tensor,
                 neg_logits: torch.Tensor,
                 batch: TrainingBatch, *args, **kwargs) -> torch.Tensor:
        tokens_passed = net.tokens_passed.to(device=pos_logits.device, dtype=torch.float32)
        epoch = tokens_passed / float(self.num_tokens_per_epoch)
        phase = epoch / float(self.num_epochs)
        return self.tau_min + torch.cos(phase * (torch.pi / 2)) * (self.tau_max - self.tau_min)

    def experiment_name(self) -> str:
        return (
            f"Cos[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g},"
            f"epochs={int(self.num_epochs)}]"
        )
    
class CosPerUserTau(Tau):
    def __call__(self, net: 'Graph',
                 pos_logits: torch.Tensor,
                 neg_logits: torch.Tensor,
                 batch: TrainingBatch, *args, **kwargs) -> torch.Tensor:
        logits = torch.concat([ 
                pos_logits.unsqueeze(2),
                neg_logits
        ], dim=-1)
        tau = ((1 + torch.cos(torch.pi * (logits.detach() + 1))) *
                (self.tau_max - self.tau_min) / 2 + self.tau_min)
        return tau

    def experiment_name(self) -> str:
        return (
            f"CosPerUser[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g}]"
        )

__all__ = ['Tau', 'ConstantTau', 'ParameterTau', 'LinearTau', 'CosTau', 'CosPerUserTau']