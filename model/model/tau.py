import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from torch import nn
from dataset import TrainingBatch

if TYPE_CHECKING:
    from .graph import Graph


class Tau(nn.Module, ABC):
    def __init__(self,
                 tau_min: float = 0.055,
                 tau_max: float = 0.06,
                 num_epochs: int = 5,
                 num_tokens_per_epoch: int = 4_019_032) -> None:
        super().__init__()
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
