"""Toy potentials from Chapter 4 of the thesis:
- 2D Shell (Eq. 24)
- 10D Gaussian (Eq. 25)
- 2D Wolfe-Quapp (Eq. 26)
"""
from __future__ import annotations

import math
import torch


class ToyPotential:
    name: str
    dim: int
    dt: float
    init_lr: float
    sigma_unit: float
    plot_lim: float

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute potential energy V(x) for x of shape (..., dim)."""
        raise NotImplementedError

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Compute force F(x) = -grad V(x) for x of shape (..., dim)."""
        raise NotImplementedError

    def init_position(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError


class Shell2D(ToyPotential):
    """2D Shell potential: V(x) = |r - r0| / alpha, r0 = sqrt(2), alpha = 0.5."""

    name = "2D Shell"
    dim = 2
    dt = 0.01
    init_lr = 0.1
    sigma_unit = 1.0  # 0.5 * r0^2 = 0.5 * 2 = 1.0
    plot_lim = 5.0

    def __init__(self, r0: float = math.sqrt(2.0), alpha: float = 0.5):
        self.r0 = r0
        self.alpha = alpha

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(x, dim=-1)
        return torch.abs(r - self.r0) / self.alpha

    def force(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(x, dim=-1, keepdim=True)
        safe_r = torch.clamp(r, min=1e-12)
        sign = torch.sign(r - self.r0)
        return -(sign / self.alpha) * (x / safe_r)

    def init_position(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        theta = 2.0 * math.pi * torch.rand(batch_size, 1, generator=generator)
        return self.r0 * torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)


class Gaussian10D(ToyPotential):
    """10D Gaussian potential: V(x) = 0.5 * sum_i (x_i^2 / alpha_i), alpha_i = 1."""

    name = "10D Gaussian"
    dim = 10
    dt = 0.1
    init_lr = 0.1
    sigma_unit = 1.0
    plot_lim = 10.0

    def __init__(self, dim: int = 10, alpha: float = 1.0):
        self.dim = dim
        self.alpha = alpha

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((x ** 2) / self.alpha, dim=-1)

    def force(self, x: torch.Tensor) -> torch.Tensor:
        return -x / self.alpha

    def init_position(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(batch_size, self.dim, generator=generator)


class WolfeQuapp2D(ToyPotential):
    """2D Wolfe-Quapp potential:
    V(x) = x1^4 + x2^4 - 2*x1^2 - 4*x2^2 + x1*x2 + 0.3*x1 + 0.1*x2
    """

    name = "2D Wolfe-Quapp"
    dim = 2
    dt = 0.1
    init_lr = 0.025
    sigma_unit = 1.0
    plot_lim = 2.5

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1]
        return (
            x1 ** 4
            + x2 ** 4
            - 2.0 * x1 ** 2
            - 4.0 * x2 ** 2
            + x1 * x2
            + 0.3 * x1
            + 0.1 * x2
        )

    def force(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1]
        fx1 = -(4.0 * x1 ** 3 - 4.0 * x1 + x2 + 0.3)
        fx2 = -(4.0 * x2 ** 3 - 8.0 * x2 + x1 + 0.1)
        return torch.stack([fx1, fx2], dim=-1)

    def init_position(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return 0.5 * torch.randn(batch_size, 2, generator=generator)


POTENTIALS = {
    "shell": Shell2D,
    "gaussian": Gaussian10D,
    "wolfe_quapp": WolfeQuapp2D,
}
