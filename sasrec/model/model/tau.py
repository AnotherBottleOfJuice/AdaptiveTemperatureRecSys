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


class ShiftedCosPerUserTau(Tau):
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
            f"ShiftedCosPerUser[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g},"
            f"shift={float(self.shift):.5g},"
            f"scale={float(self.scale):.5g}]"
        )


class ExpandedShiftedCosPerUserTau(Tau):
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
        # clamp both ends: clamp_min(0) gives the low-logit tau_max plateau,
        # clamp_max(2*scale) gives the high-logit tau_max plateau (without it the
        # cosine re-enters its next period and dips back to tau_min near logit=1).
        bounded_logits = shifted_logits.clamp(min=0.0, max=2.0 * self.scale)
        delta_tau = self.tau_max - self.tau_min
        tau = self.tau_min + 0.5 * delta_tau * (
            1 + torch.cos(bounded_logits * torch.pi / self.scale)
        )
        return tau

    def experiment_name(self) -> str:
        return (
            f"ExpandedShiftedCosPerUser[min={float(self.tau_min):.5g},"
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


class ConfidenceShiftedCosPerUserTau(Tau):
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
        conf_min: float,
        conf_max: float,
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
        if conf_max <= conf_min:
            raise ValueError("conf_max must be greater than conf_min")
        self.min_shift = float(min_shift)
        self.max_shift = float(max_shift)
        self.scale = float(scale)
        self.conf_min = float(conf_min)
        self.conf_max = float(conf_max)

    def shift_for(self, confidence: Tensor | float) -> Tensor:
        confidence = torch.as_tensor(confidence, dtype=torch.float32)
        phase = ((confidence - self.conf_min) / (self.conf_max - self.conf_min)).clamp(
            0.0, 1.0
        )
        return self.min_shift + phase * (self.max_shift - self.min_shift)

    def tau_for(self, logits: Tensor, confidence: Tensor | float) -> Tensor:
        shift = self.shift_for(confidence).to(logits.device)
        # clamp both ends so high logits plateau at tau_max instead of the cosine
        # wrapping into its next period (see ExpandedShiftedCosPerUserTau).
        bounded_logits = (shift + logits).clamp(min=0.0, max=2.0 * self.scale)
        delta_tau = self.tau_max - self.tau_min
        return self.tau_min + 0.5 * delta_tau * (
            1 + torch.cos(bounded_logits * torch.pi / self.scale)
        )

    def __call__(self, ctx: TauContext) -> Tensor:
        logits = torch.concat(
            [ctx.pos_logits.unsqueeze(2), ctx.neg_logits], dim=-1
        ).detach()
        confidence = ctx.pos_logits.mean().detach()
        return self.tau_for(logits, confidence)

    def experiment_name(self) -> str:
        return (
            f"ConfidenceShiftedCosPerUser[min={float(self.tau_min):.5g},"
            f"max={float(self.tau_max):.5g},"
            f"min_shift={float(self.min_shift):.5g},"
            f"max_shift={float(self.max_shift):.5g},"
            f"scale={float(self.scale):.5g},"
            f"conf=[{float(self.conf_min):.5g},{float(self.conf_max):.5g}]]"
        )


class DoubleDipCosPerUserTau(Tau):
    """Shifted cosine per-user, but with an extra sharpening for near-best logits.

    As the (shifted) logit ``u = shift + logit`` grows from 0 over three ``scale``
    wide cosine segments, tau follows:

        tau_max  ->  tau_min  ->  tau_mid  ->  tau_low

    i.e. it dips to the sharpest tau (tau_min) for typical positives, relaxes back
    to an intermediate tau_mid, then dips again to tau_low (lower than tau_mid but
    not as low as tau_min) for the most confident / near-best logits. ``u`` is
    clamped to ``[0, 3 * scale]`` so low logits stay at tau_max and very high logits
    stay at tau_low. Requires ``tau_min <= tau_low <= tau_mid <= tau_max``.
    """

    def __init__(
        self,
        initial_tau: float,
        tau_min: float,
        tau_max: float,
        num_epochs: int,
        num_tokens_per_epoch: int,
        shift: float,
        scale: float,
        tau_mid: float,
        tau_low: float,
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
        if not (tau_min <= tau_low <= tau_mid <= tau_max):
            raise ValueError("require tau_min <= tau_low <= tau_mid <= tau_max")
        self.shift = float(shift)
        self.scale = float(scale)
        self.tau_mid = float(tau_mid)
        self.tau_low = float(tau_low)

    def tau_for(self, logits: Tensor) -> Tensor:
        s = self.scale
        u = (self.shift + logits).clamp(min=0.0, max=3.0 * s)
        # seg1 (u in [0, s]):    tau_max -> tau_min
        t1 = self.tau_min + 0.5 * (self.tau_max - self.tau_min) * (
            1 + torch.cos(u * torch.pi / s)
        )
        # seg2 (u in [s, 2s]):   tau_min -> tau_mid
        t2 = self.tau_min + 0.5 * (self.tau_mid - self.tau_min) * (
            1 - torch.cos((u - s) * torch.pi / s)
        )
        # seg3 (u in [2s, 3s]):  tau_mid -> tau_low
        t3 = self.tau_low + 0.5 * (self.tau_mid - self.tau_low) * (
            1 + torch.cos((u - 2 * s) * torch.pi / s)
        )
        return torch.where(u <= s, t1, torch.where(u <= 2 * s, t2, t3))

    def __call__(self, ctx: TauContext) -> Tensor:
        logits = torch.concat(
            [ctx.pos_logits.unsqueeze(2), ctx.neg_logits], dim=-1
        ).detach()
        return self.tau_for(logits)

    def experiment_name(self) -> str:
        return (
            f"DoubleDipCosPerUser[min={float(self.tau_min):.5g},"
            f"low={float(self.tau_low):.5g},"
            f"mid={float(self.tau_mid):.5g},"
            f"max={float(self.tau_max):.5g},"
            f"shift={float(self.shift):.5g},"
            f"scale={float(self.scale):.5g}]"
        )


__all__ = [
    "TauContext",
    "Tau",
    "ConstantTau",
    "ParameterTau",
    "LinearTau",
    "CosTau",
    "CosPerUserTau",
    "ShiftedCosPerUserTau",
    "ExpandedShiftedCosPerUserTau",
    "MACLossTau",
    "ConfidenceShiftedCosPerUserTau",
    "DoubleDipCosPerUserTau",
]
