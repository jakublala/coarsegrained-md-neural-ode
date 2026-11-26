"""Rigid-body quaternion dynamics (Richardson Velocity-Verlet) and three CG potential architectures:
  1. BaselineCGNet:    Paper's 68-48-32 tanh MLP on raw 11D [dx, q1, q2] (no regularization, no cutoff)
  2. RegularizedCGNet: Same raw 11D MLP trained with L2 weight decay + temporal regularization (STEER)
  3. ImprovedCGNet:    SO(3)-invariant, q==-q invariant, 1<->2 symmetric MLP with C2 smooth cutoff envelope
                       and non-negative (softplus) output, trained with collision-weighted batches and
                       normalized multi-step + sign-invariant quaternion loss.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


RC_LJ = 2.0 ** (1.0 / 6.0)  # 1.1224620483
R_MAX = 2.0 + RC_LJ          # 3.1224620483
R_ON = 2.60

# 7-particle regular hexagon coordinates in LAMMPS body-fixed principal-axis frame
# (l = 1.0, in xy-plane; LAMMPS assigns q0 = [0, 1/sqrt(2), 1/sqrt(2), 0] at t=0,
# which rotates hex.conf by 90 deg -> vertices at 30, 90, 150, 210, 270, 330 deg)
_angles = torch.arange(6, dtype=torch.float32) * (math.pi / 3.0) + (math.pi / 2.0)
HEX_LOCAL = torch.cat([
    torch.zeros(1, 3),
    torch.stack([torch.cos(_angles), torch.sin(_angles), torch.zeros(6)], dim=-1),
], dim=0)  # (7, 3)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion q = [w, x, y, z] (..., 4) -> rotation matrix R (..., 3, 3)."""
    w, x, y, z = q.unbind(dim=-1)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


def quat_E(q: torch.Tensor) -> torch.Tensor:
    """Lab-frame quaternion Jacobian matrix E(q) of shape (..., 3, 4) such that
    dq/dt = 0.5 * E(q)^T omega_lab  and  tau_lab = -0.5 * E(q) (dV/dq).
    """
    w, x, y, z = q.unbind(dim=-1)
    row0 = torch.stack([-x, w, -z, y], dim=-1)
    row1 = torch.stack([-y, z, w, -x], dim=-1)
    row2 = torch.stack([-z, -y, x, w], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def smooth_cutoff(r: torch.Tensor, r_on: float = R_ON, r_max: float = R_MAX) -> torch.Tensor:
    """C^2 smooth quintic cutoff switch: 1 for r <= r_on, 0 for r >= r_max."""
    u = ((r - r_on) / (r_max - r_on)).clamp(0.0, 1.0)
    return 1.0 - (u ** 3) * (10.0 - 15.0 * u + 6.0 * (u ** 2))


def exact_hex_potential(dx: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Exact ground-truth all-atom WCA potential between two 7-particle hexagons.
    dx = x2 - x1 of shape (B, 3); q1, q2 of shape (B, 4). Returns V of shape (B,).
    """
    device = dx.device
    pts = HEX_LOCAL.to(device)  # (7, 3)
    R1 = quat_to_rotmat(q1)     # (B, 3, 3)
    R2 = quat_to_rotmat(q2)     # (B, 3, 3)
    p1 = torch.einsum("bij,nj->bni", R1, pts)           # (B, 7, 3)
    p2 = torch.einsum("bij,nj->bni", R2, pts) + dx[:, None, :]  # (B, 7, 3)
    diff = p2[:, None, :, :] - p1[:, :, None, :]        # (B, 7, 7, 3)
    r2 = (diff ** 2).sum(dim=-1)                        # (B, 7, 7)
    mask = r2 < (RC_LJ ** 2)
    safe_r2 = torch.where(mask, r2.clamp(min=1e-4), torch.ones_like(r2))
    inv_r6 = 1.0 / (safe_r2 ** 3)
    v_pair = torch.where(mask, 4.0 * (inv_r6 ** 2 - inv_r6) + 1.0, torch.zeros_like(r2))
    return v_pair.sum(dim=(-1, -2))


def external_restraint_potential(dx: torch.Tensor, K_ext: float, r0_ext: float = 2.0) -> torch.Tensor:
    """Harmonic restraint V_ext(r) = 0.5 * K_ext * (||dx|| - r0)^2 (Eq. 21)."""
    r = torch.linalg.norm(dx, dim=-1)
    return 0.5 * K_ext * ((r - r0_ext) ** 2)


class RawMLPNet(nn.Module):
    """Paper baseline architecture: 11 -> 68 -> 48 -> 32 -> 1 with tanh activations."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 68),
            nn.Tanh(),
            nn.Linear(68, 48),
            nn.Tanh(),
            nn.Linear(48, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, dx: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([dx, q1, q2], dim=-1)
        return self.net(inp).squeeze(-1)


class ImprovedCGNet(nn.Module):
    """Physics-informed CG potential with:
    1. Exact SO(3) rotational invariance + quaternion double-cover (q == -q) invariance via R(q)
    2. Exact particle-exchange symmetry (body 1 <-> body 2)
    3. C^2 smooth cutoff envelope at r_max = 2l + r_c and non-negative softplus output
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        # Invariant feature dimension:
        #   r (1) + u12 = R1^T dx (3) + u21 = R2^T (-dx) (3) + R12 = R1^T R2 (9) = 16
        # Plus hexagonal symmetry features: we also include r_xy_1, z1^2, r_xy_2, z2^2, etc.
        # Keep it general for any rigid body (16 general SE(3) invariants):
        self.phi = nn.Sequential(
            nn.Linear(16, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        for m in self.phi:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _features(self, dx: torch.Tensor, R1: torch.Tensor, R2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = torch.linalg.norm(dx, dim=-1, keepdim=True).clamp(min=1e-6)
        u12 = torch.einsum("bij,bi->bj", R1, dx)         # R1^T dx
        u21 = torch.einsum("bij,bi->bj", R2, -dx)        # R2^T (-dx)
        R12 = torch.matmul(R1.transpose(-1, -2), R2)     # (B, 3, 3)
        R21 = R12.transpose(-1, -2)
        f12 = torch.cat([r, u12, u21, R12.reshape(-1, 9)], dim=-1)
        f21 = torch.cat([r, u21, u12, R21.reshape(-1, 9)], dim=-1)
        return r.squeeze(-1), f12, f21

    def forward(self, dx: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        R1 = quat_to_rotmat(q1)
        R2 = quat_to_rotmat(q2)
        r, f12, f21 = self._features(dx, R1, R2)
        raw = 0.5 * (self.phi(f12) + self.phi(f21)).squeeze(-1)
        env = smooth_cutoff(r, R_ON, R_MAX)
        return env * F.softplus(raw)


def compute_forces_and_torques(
    pot_fn,
    dx: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    K_ext: float,
    r0_ext: float,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute F_on_2 = -dV/d(dx) (so F_on_1 = -F_on_2), tau1_lab, tau2_lab."""
    dx_req = dx.detach().requires_grad_(True) if not dx.requires_grad else dx
    q1_req = q1.detach().requires_grad_(True) if not q1.requires_grad else q1
    q2_req = q2.detach().requires_grad_(True) if not q2.requires_grad else q2

    v_cg = pot_fn(dx_req, q1_req, q2_req)
    v_ext = external_restraint_potential(dx_req, K_ext, r0_ext)
    v_tot = (v_cg + v_ext).sum()

    g_dx, g_q1, g_q2 = torch.autograd.grad(
        v_tot, (dx_req, q1_req, q2_req), create_graph=create_graph, allow_unused=True
    )
    if g_dx is None:
        g_dx = torch.zeros_like(dx_req)
    if g_q1 is None:
        g_q1 = torch.zeros_like(q1_req)
    if g_q2 is None:
        g_q2 = torch.zeros_like(q2_req)
    # Since dx = x2 - x1: F2 = -dV/dx2 = -g_dx, F1 = -dV/dx1 = +g_dx
    F2 = -g_dx
    tau1 = -0.5 * torch.einsum("bij,bj->bi", quat_E(q1_req), g_q1)
    tau2 = -0.5 * torch.einsum("bij,bj->bi", quat_E(q2_req), g_q2)
    return F2, tau1, tau2


def _qdot(q: torch.Tensor, L_lab: torch.Tensor, J_inv: torch.Tensor) -> torch.Tensor:
    R = quat_to_rotmat(q)
    l_body = torch.einsum("bij,bi->bj", R, L_lab)          # R^T L_lab
    omega_body = l_body * J_inv                            # J^{-1} l_body
    omega_lab = torch.einsum("bij,bj->bi", R, omega_body)  # R omega_body
    return 0.5 * torch.einsum("bji,bj->bi", quat_E(q), omega_lab)


def _richardson_quat_step(q: torch.Tensor, L_lab: torch.Tensor, J_inv: torch.Tensor, dt: float) -> torch.Tensor:
    """LAMMPS Richardson-iterated quaternion free-rotation step (Eqs. 14-17)."""
    qdot0 = _qdot(q, L_lab, J_inv)
    q_full = F.normalize(q + dt * qdot0, dim=-1)
    q_half_mid = F.normalize(q + 0.5 * dt * qdot0, dim=-1)
    qdot_mid = _qdot(q_half_mid, L_lab, J_inv)
    q_half_end = F.normalize(q_half_mid + 0.5 * dt * qdot_mid, dim=-1)
    return F.normalize(2.0 * q_half_end - q_full, dim=-1)


def rollout_rigid_body(
    pot_fn,
    dx0: torch.Tensor,
    dv0: torch.Tensor,
    q1_0: torch.Tensor,
    q2_0: torch.Tensor,
    L1_0: torch.Tensor,
    L2_0: torch.Tensor,
    dt: float,
    n_steps: int,
    K_ext: float,
    r0_ext: float = 2.0,
    mass: float = 7.0,
    inertia: tuple[float, float, float] = (3.0, 3.0, 6.0),
    create_graph: bool = True,
    record_all: bool = True,
) -> dict[str, torch.Tensor]:
    """Roll out relative translational (dx = x2 - x1, dv = v2 - v1) and rotational
    (q1, q2, L1, L2) rigid-body dynamics for n_steps using Richardson Velocity-Verlet.

    Note: In the two-body CM frame, d(dx)/dt = dv, and d(dv)/dt = F2/m2 - F1/m1 = (2/m) F2.
    """
    J_inv = torch.tensor([1.0 / inertia[0], 1.0 / inertia[1], 1.0 / inertia[2]], device=dx0.device, dtype=dx0.dtype)
    inv_mu = 2.0 / mass  # reduced-mass inverse for identical bodies

    dx, dv = dx0, dv0
    q1, q2 = q1_0, q2_0
    L1, L2 = L1_0, L2_0

    F2, tau1, tau2 = compute_forces_and_torques(pot_fn, dx, q1, q2, K_ext, r0_ext, create_graph=create_graph)

    if record_all:
        dxs, dvs, q1s, q2s, L1s, L2s = [dx], [dv], [q1], [q2], [L1], [L2]

    half_dt = 0.5 * dt
    for _ in range(n_steps):
        # 1. Half-kick
        dv_half = dv + half_dt * (inv_mu * F2)
        L1_half = L1 + half_dt * tau1
        L2_half = L2 + half_dt * tau2

        # 2. Translational drift
        dx = dx + dt * dv_half

        # 3. Rotational drift (Richardson quaternion leapfrog)
        q1 = _richardson_quat_step(q1, L1_half, J_inv, dt)
        q2 = _richardson_quat_step(q2, L2_half, J_inv, dt)

        if not create_graph:
            dx = dx.detach()
            dv_half = dv_half.detach()
            q1 = q1.detach()
            q2 = q2.detach()
            L1_half = L1_half.detach()
            L2_half = L2_half.detach()

        # 4. Recompute forces/torques and complete half-kick
        F2, tau1, tau2 = compute_forces_and_torques(pot_fn, dx, q1, q2, K_ext, r0_ext, create_graph=create_graph)
        dv = dv_half + half_dt * (inv_mu * F2)
        L1 = L1_half + half_dt * tau1
        L2 = L2_half + half_dt * tau2

        if not create_graph:
            dv = dv.detach()
            L1 = L1.detach()
            L2 = L2.detach()

        if record_all:
            dxs.append(dx)
            dvs.append(dv)
            q1s.append(q1)
            q2s.append(q2)
            L1s.append(L1)
            L2s.append(L2)

    if record_all:
        return {
            "dx": torch.stack(dxs, dim=1),
            "dv": torch.stack(dvs, dim=1),
            "q1": torch.stack(q1s, dim=1),
            "q2": torch.stack(q2s, dim=1),
            "L1": torch.stack(L1s, dim=1),
            "L2": torch.stack(L2s, dim=1),
        }
    return {"dx": dx, "dv": dv, "q1": q1, "q2": q2, "L1": L1, "L2": L2}
