"""Physics-informed Neural ODE potential (2-layer, 50-neuron Sigmoid MLP)
with symplectic Velocity-Verlet integration and analytical force computation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PotentialNet(nn.Module):
    """2-hidden-layer MLP with Sigmoid activations and linear scalar output,
    initialized with Glorot (Xavier) initialization as in Ch. 3 & 4.
    """

    def __init__(self, dim: int, hidden_dim: int = 50):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scalar potential V'(x) for x of shape (..., dim) -> (...)."""
        h1 = torch.sigmoid(self.fc1(x))
        h2 = torch.sigmoid(self.fc2(h1))
        return self.fc3(h2).squeeze(-1)

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Exact analytical force F(x) = -dV'/dx, differentiable w.r.t. network weights."""
        h1 = torch.sigmoid(self.fc1(x))
        h2 = torch.sigmoid(self.fc2(h1))
        # dV/dh2 = W3 (shape 1, H)
        # dV/da2 = dV/dh2 * h2 * (1 - h2)
        d_a2 = self.fc3.weight * (h2 * (1.0 - h2))  # (..., H)
        # dV/dh1 = d_a2 @ W2
        d_h1 = d_a2 @ self.fc2.weight  # (..., H)
        d_a1 = d_h1 * (h1 * (1.0 - h1))  # (..., H)
        grad_x = d_a1 @ self.fc1.weight  # (..., dim)
        return -grad_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy(x)


def rollout_neural_ode(
    model: PotentialNet,
    x0: torch.Tensor,
    p0: torch.Tensor,
    dt: float,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Velocity-Verlet rollout for n_steps.
    Returns (X_hat, P_hat) of shape (batch, n_steps + 1, dim).
    """
    x = x0
    p = p0
    f = model.force(x)
    xs = [x]
    ps = [p]
    half_dt = 0.5 * dt
    for _ in range(n_steps):
        p_half = p + half_dt * f
        x = x + dt * p_half
        f = model.force(x)
        p = p_half + half_dt * f
        xs.append(x)
        ps.append(p)
    return torch.stack(xs, dim=1), torch.stack(ps, dim=1)


def compute_loss(
    x_true: torch.Tensor,
    x_pred: torch.Tensor,
    p_true: torch.Tensor,
    p_pred: torch.Tensor,
    loss_type: str = "both",
) -> torch.Tensor:
    """Loss functions from Eqs. 27-29 (unit-normalized MAE)."""
    if loss_type == "x":
        return torch.mean(torch.abs(x_true - x_pred))
    elif loss_type == "p":
        return torch.mean(torch.abs(p_true - p_pred))
    elif loss_type == "both":
        return 0.5 * (
            torch.mean(torch.abs(x_true - x_pred))
            + torch.mean(torch.abs(p_true - p_pred))
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def train_model(
    X_train: torch.Tensor,
    P_train: torch.Tensor,
    X_test: torch.Tensor,
    P_test: torch.Tensor,
    dt: float,
    init_lr: float,
    epochs: int = 5000,
    batch_size: int = 800,
    rollout_steps: int = 10,
    loss_type: str = "both",
    sched_freq: int = 2000,
    sched_factor: float = 0.6,
    seed: int = 0,
    record_history: bool = False,
) -> dict:
    """Train a PotentialNet on trajectory snippets of length rollout_steps."""
    torch.manual_seed(seed)
    dim = X_train.shape[-1]
    model = PotentialNet(dim=dim, hidden_dim=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: sched_factor ** (ep // sched_freq)
    )

    n_train, total_steps, _ = X_train.shape
    max_start = total_steps - rollout_steps - 1
    actual_batch = min(batch_size, n_train)

    gen = torch.Generator()
    gen.manual_seed(seed + 1000)

    loss_history = []
    for ep in range(epochs):
        # Sample random trajectories and random start times within each trajectory
        traj_idx = torch.randint(0, n_train, (actual_batch,), generator=gen)
        t0_idx = torch.randint(0, max_start + 1, (actual_batch,), generator=gen)
        # Gather snippets of length rollout_steps + 1
        t_offsets = torch.arange(rollout_steps + 1).unsqueeze(0)  # (1, T+1)
        t_mat = t0_idx.unsqueeze(1) + t_offsets  # (B, T+1)
        x_batch = X_train[traj_idx.unsqueeze(1), t_mat]  # (B, T+1, D)
        p_batch = P_train[traj_idx.unsqueeze(1), t_mat]  # (B, T+1, D)

        optimizer.zero_grad(set_to_none=True)
        x_hat, p_hat = rollout_neural_ode(model, x_batch[:, 0], p_batch[:, 0], dt, rollout_steps)
        loss = compute_loss(x_batch, x_hat, p_batch, p_hat, loss_type=loss_type)
        loss.backward()
        # Mild gradient clipping prevents NaNs while preserving Wolfe-Quapp dynamics
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        scheduler.step()

        if record_history and (ep % 50 == 0 or ep == epochs - 1):
            loss_history.append((ep, float(loss.item())))

    # Evaluate on test set using full combined loss L(X, X_hat, P, P_hat) over rollout_steps
    with torch.no_grad():
        x_t = X_test[:, : rollout_steps + 1, :]
        p_t = P_test[:, : rollout_steps + 1, :]
        x_hat_t, p_hat_t = rollout_neural_ode(model, x_t[:, 0], p_t[:, 0], dt, rollout_steps)
        test_loss = float(compute_loss(x_t, x_hat_t, p_t, p_hat_t, loss_type="both").item())

    return {
        "model": model,
        "test_loss": test_loss,
        "loss_history": loss_history,
    }
