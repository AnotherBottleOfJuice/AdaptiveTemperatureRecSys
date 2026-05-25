import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from torch import nn, Tensor
from yambdadataset import TrainingBatch

if TYPE_CHECKING:
    from .graph import Graph


@dataclass
class TauContext:
    net: "Graph"
    pos_logits: Tensor
    neg_logits: Tensor
    batch: TrainingBatch


class Tau(nn.Module, ABC):
    def __init__(
        self,
        initial_tau: float,
        tau_min: float,
        tau_max: float,
        num_epochs: int,
        num_tokens_per_epoch: int,
    ) -> None:
        super().__init__()
        self.initial_tau = initial_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.num_epochs = num_epochs
        self.num_tokens_per_epoch = num_tokens_per_epoch

    @abstractmethod
    def __call__(self, ctx: TauContext) -> Tensor:
        pass

    @abstractmethod
    def experiment_name(self) -> str:
        pass


class ConstantTau(Tau):
    def __call__(self, ctx: TauContext) -> Tensor:
        return torch.tensor(
            self.initial_tau, device=ctx.pos_logits.device, dtype=torch.float32
        )

    def experiment_name(self) -> str:
        return f"Constant[value={self.initial_tau:.5g}]"


class ParameterTau(Tau):
    def __init__(
        self,
        initial_tau: float,
        tau_min: float,
        tau_max: float,
        num_epochs: int,
        num_tokens_per_epoch: int,
    ) -> None:
        super().__init__(
            initial_tau=initial_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            num_epochs=num_epochs,
            num_tokens_per_epoch=num_tokens_per_epoch,
        )
        self.initial_tau = float(initial_tau)
        self.tau = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def __call__(self, ctx: TauContext) -> Tensor:
        tau_normalized = torch.sigmoid(self.tau)
        tau = self.tau_min + tau_normalized * (self.tau_max - self.tau_min)
        return tau.to(ctx.pos_logits.device)

    def experiment_name(self) -> str:
        return f"Param[init={self.initial_tau:.5g}]"


class LinearTau(Tau):
    def __call__(self, ctx: TauContext) -> Tensor:
        tokens_passed = ctx.net.tokens_passed.to(
            device=ctx.pos_logits.device, dtype=torch.float32
        )
        phase = tokens_passed / (
            float(self.num_tokens_per_epoch) * float(self.num_epochs)
        )
        return self.tau_max + phase * (self.tau_min - self.tau_max)

    def experiment_name(self) -> str:
        return (
            f"Linear[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g},"
            f"epochs={int(self.num_epochs)}]"
        )


class CosTau(Tau):
    def __call__(self, ctx: TauContext) -> Tensor:
        tokens_passed = ctx.net.tokens_passed.to(
            device=ctx.pos_logits.device, dtype=torch.float32
        )
        epoch = tokens_passed / float(self.num_tokens_per_epoch)
        phase = epoch / float(self.num_epochs)
        return self.tau_min + torch.cos(phase * (torch.pi / 2)) * (
            self.tau_max - self.tau_min
        )

    def experiment_name(self) -> str:
        return (
            f"Cos[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g},"
            f"epochs={int(self.num_epochs)}]"
        )


class CosPerUserTau(Tau):
    def __call__(self, ctx: TauContext) -> Tensor:
        logits = torch.concat([ctx.pos_logits.unsqueeze(2), ctx.neg_logits], dim=-1)
        tau = (1 + torch.cos(torch.pi * (logits.detach() + 1))) * (
            self.tau_max - self.tau_min
        ) / 2 + self.tau_min
        return tau

    def experiment_name(self) -> str:
        return (
            f"CosPerUser[min={float(self.tau_min):.5g},max={float(self.tau_max):.5g}]"
        )


class ShifetedCosPerUserTau(Tau):
    def __init__(
        self,
        initial_tau: float,
        tau_min: float,
        tau_max: float,
        num_epochs: int,
        num_tokens_per_epoch: int,
        shift: float,
        scale: float,
    ) -> None:
        super().__init__(
            initial_tau=initial_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            num_epochs=num_epochs,
            num_tokens_per_epoch=num_tokens_per_epoch,
        )
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.shift = float(shift)
        self.scale = float(scale)

    def __call__(self, ctx: TauContext) -> Tensor:
        logits = torch.concat(
            [ctx.pos_logits.unsqueeze(2), ctx.neg_logits], dim=-1
        ).detach()
        shifted_logits = self.shift + logits
        bounded_logits = (
            shifted_logits.clamp_max(0.0)
            if self.shift <= 0
            else shifted_logits.clamp_min(0.0)
        ).clamp(min=-2 * self.scale, max=2 * self.scale)
        delta_tau = self.tau_max - self.tau_min
        tau = self.tau_min + 0.5 * delta_tau * (
            1 + torch.cos(bounded_logits * torch.pi / self.scale)
        )
        return tau

    def experiment_name(self) -> str:
        return (
            f"ShifetedCosPerUser[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g},"
            f"shift={float(self.shift):.5g},"
            f"scale={float(self.scale):.5g}]"
        )


class ExpandedShifetedCosPerUserTau(Tau):
    def __init__(
        self,
        initial_tau: float,
        tau_min: float,
        tau_max: float,
        num_epochs: int,
        num_tokens_per_epoch: int,
        min_shift: float,
        max_shift: float,
        scale: float,
    ) -> None:
        super().__init__(
            initial_tau=initial_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            num_epochs=num_epochs,
            num_tokens_per_epoch=num_tokens_per_epoch,
        )
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.min_shift = float(min_shift)
        self.max_shift = float(max_shift)
        self.scale = float(scale)

    def __call__(self, ctx: TauContext) -> Tensor:
        phase = ctx.net.tokens_passed / (
            float(self.num_tokens_per_epoch) * float(self.num_epochs)
        )
        shift = self.min_shift + phase * (self.max_shift - self.min_shift)

        logits = torch.concat(
            [ctx.pos_logits.unsqueeze(2), ctx.neg_logits], dim=-1
        ).detach()
        shifted_logits = shift + logits
        bounded_logits = shifted_logits.clamp_min(0.0)
        delta_tau = self.tau_max - self.tau_min
        tau = self.tau_min + 0.5 * delta_tau * (
            1 + torch.cos(bounded_logits * torch.pi / self.scale)
        )
        return tau

    def experiment_name(self) -> str:
        return (
            f"ExpandedShifetedCosPerUser[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g},"
            f"min_shift={float(self.min_shift):.5g},"
            f"max_shift={float(self.max_shift):.5g},"
            f"scale={float(self.scale):.5g}]"
        )


class MACLossTau(Tau):
    def __init__(
        self,
        initial_tau: float,
        tau_min: float,
        tau_max: float,
        num_epochs: int,
        num_tokens_per_epoch: int,
        base_threshold: float,
        linear_coeff: float,
    ) -> None:
        super().__init__(
            initial_tau=initial_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            num_epochs=num_epochs,
            num_tokens_per_epoch=num_tokens_per_epoch,
        )
        self.base_threshold = float(base_threshold)
        self.linear_coeff = float(linear_coeff)

    def __call__(self, ctx: TauContext) -> Tensor:
        tau = (
            (ctx.pos_logits.mean().detach() - self.base_threshold) * self.linear_coeff
            + self.initial_tau
        ).clamp(self.tau_min, self.tau_max)
        return tau

    def experiment_name(self) -> str:
        return (
            f"MACLoss[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g}]"
        )


__all__ = [
    "TauContext",
    "Tau",
    "ConstantTau",
    "ParameterTau",
    "LinearTau",
    "CosTau",
    "CosPerUserTau",
    "ShifetedCosPerUserTau",
    "ExpandedShifetedCosPerUserTau",
]
