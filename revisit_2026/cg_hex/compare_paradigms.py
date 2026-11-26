"""Apples-to-apples comparison of training paradigms on the SAME SO(3)+D6+C2-cutoff CGNet:
  1. Static Energy Matching:       supervised on (dx, q1, q2) -> V_exact
  2. Static Force+Torque Matching: supervised on (dx, q1, q2) -> (F2_exact, tau1_exact, tau2_exact)
  3. Neural ODE (Trajectory):      supervised ONLY on LAMMPS trajectories (no V/F/tau labels!)

Plus a direct wall-clock time & peak-memory scaling benchmark across rollout horizon T in [10, 20, 40, 80]
comparing Static vs. Neural ODE (BPTT / O(T) memory) vs. Neural ODE (Symplectic Adjoint / O(1) memory).
"""
from __future__ import annotations

import gc
import json
import math
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rigid_ode import (
    quat_to_rotmat,
    quat_E,
    smooth_cutoff,
    exact_hex_potential,
    compute_forces_and_torques,
    rollout_rigid_body,
    _richardson_quat_step,
    R_MAX,
    R_ON,
)
from train_compare import prepare_splits, sample_batch, set_style

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hex_dataset.pt"
OUT_DIR = ROOT / "results"
FIG_DIR = ROOT.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def _cos6_smooth(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Smooth D6-invariant feature rho * cos(6 * atan2(y, x)) without atan2 singularity at rho=0."""
    r2 = x * x + y * y
    rho = torch.sqrt(r2 + 1e-8)
    # Re((x + i y)^6) = x^6 - 15 x^4 y^2 + 15 x^2 y^4 - y^6
    re6 = (
        x ** 6
        - 15.0 * (x ** 4) * (y ** 2)
        + 15.0 * (x ** 2) * (y ** 4)
        - y ** 6
    )
    return re6 / (rho ** 5 + 1e-6)


class D6SymmetricCGNet(nn.Module):
    """SO(3)-invariant, q==-q invariant, D6 (6-fold + z-flip) hexagonal-invariant,
    1<->2 exchange-symmetric CG potential with C^2 smooth cutoff and steep softplus^2 output.
    """

    def __init__(self, hidden: int = 96):
        super().__init__()
        # 12 smooth SO(3)+D6 invariants per ordered pair (1, 2):
        #   r, 1/(r+0.1), rho1, z1^2, cos6(u12), rho2, z2^2, cos6(u21),
        #   (n1.n2)^2, rho_n2_in_1, cos6(n2_in_1), (u12_xy . n2_xy)^2
        self.phi = nn.Sequential(
            nn.Linear(12, hidden),
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

    def _pair_features(self, r: torch.Tensor, u12: torch.Tensor, u21: torch.Tensor, R12: torch.Tensor) -> torch.Tensor:
        # u12 = R1^T dx (pos of 2 in 1's frame); n2_in_1 = R12[:, :, 2] (normal of 2 in 1's frame)
        rho1 = torch.sqrt(u12[:, 0] ** 2 + u12[:, 1] ** 2 + 1e-8)
        z1_sq = u12[:, 2] ** 2
        c6_1 = _cos6_smooth(u12[:, 0], u12[:, 1])

        rho2 = torch.sqrt(u21[:, 0] ** 2 + u21[:, 1] ** 2 + 1e-8)
        z2_sq = u21[:, 2] ** 2
        c6_2 = _cos6_smooth(u21[:, 0], u21[:, 1])

        n2_in_1 = R12[:, :, 2]  # (B, 3)
        n1_dot_n2_sq = n2_in_1[:, 2] ** 2
        rho_n2 = torch.sqrt(n2_in_1[:, 0] ** 2 + n2_in_1[:, 1] ** 2 + 1e-8)
        c6_n2 = _cos6_smooth(n2_in_1[:, 0], n2_in_1[:, 1])
        # Alignment between in-plane separation and projected normal of body 2 (z-flip invariant via square)
        proj_align_sq = (u12[:, 0] * n2_in_1[:, 0] + u12[:, 1] * n2_in_1[:, 1]) ** 2

        return torch.stack([
            r,
            1.0 / (r + 0.1),
            rho1,
            z1_sq,
            c6_1,
            rho2,
            z2_sq,
            c6_2,
            n1_dot_n2_sq,
            rho_n2,
            c6_n2,
            proj_align_sq,
        ], dim=-1)

    def forward(self, dx: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        R1 = quat_to_rotmat(q1)
        R2 = quat_to_rotmat(q2)
        r = torch.linalg.norm(dx, dim=-1).clamp(min=1e-6)
        u12 = torch.einsum("bij,bi->bj", R1, dx)
        u21 = torch.einsum("bij,bi->bj", R2, -dx)
        R12 = torch.matmul(R1.transpose(-1, -2), R2)
        R21 = R12.transpose(-1, -2)

        f12 = self._pair_features(r, u12, u21, R12)
        f21 = self._pair_features(r, u21, u12, R21)
        raw = 0.5 * (self.phi(f12) + self.phi(f21)).squeeze(-1)
        env = smooth_cutoff(r, R_ON, R_MAX)
        # Quadratic softplus^2 easily represents steep r^-12 walls while staying smooth and >= 0
        return env * (F.softplus(raw) ** 2)


class ReversibleVerletStep(torch.autograd.Function):
    """True O(1)-memory symplectic adjoint step for Velocity-Verlet!
    Forward: runs T Velocity-Verlet steps WITHOUT saving intermediate computation graphs (O(1) memory).
    Backward: reverses the symplectic integrator step-by-step, rebuilding only 1 step's local graph at a time
              to accumulate gradients wrt (dx, dv, q1, q2, L1, L2, *params) — exact O(1) memory in T!
    """

    @staticmethod
    def forward(ctx, dx, dv, q1, q2, L1, L2, dt, n_steps, K_ext, r0_ext, mass, pot_fn, *params):
        ctx.dt = dt
        ctx.n_steps = n_steps
        ctx.K_ext = K_ext
        ctx.r0_ext = r0_ext
        ctx.mass = mass
        ctx.pot_fn = pot_fn
        ctx.n_params = len(params)

        with torch.no_grad():
            with torch.enable_grad():
                out = rollout_rigid_body(
                    pot_fn, dx.detach(), dv.detach(), q1.detach(), q2.detach(), L1.detach(), L2.detach(),
                    dt=dt, n_steps=n_steps, K_ext=K_ext, r0_ext=r0_ext, mass=mass,
                    create_graph=False, record_all=False,
                )
        ctx.save_for_backward(out["dx"], out["dv"], out["q1"], out["q2"], out["L1"], out["L2"], *params)
        return out["dx"], out["dv"], out["q1"], out["q2"], out["L1"], out["L2"]

    @staticmethod
    def backward(ctx, g_dx, g_dv, g_q1, g_q2, g_L1, g_L2):
        saved = ctx.saved_tensors
        dx, dv, q1, q2, L1, L2 = [t.detach() for t in saved[:6]]
        params = saved[6:]
        dt = ctx.dt
        K_ext, r0_ext, mass = ctx.K_ext, ctx.r0_ext, ctx.mass
        pot_fn = ctx.pot_fn

        grad_params = [torch.zeros_like(p) for p in params]
        adjs = (g_dx, g_dv, g_q1, g_q2, g_L1, g_L2)

        # Reverse T steps one at a time (only 1 step graph alive in memory at any instant!)
        for _ in range(ctx.n_steps):
            # Reconstruct previous state by stepping backward (-dt) without graph
            with torch.enable_grad():
                prev = rollout_rigid_body(
                    pot_fn, dx, dv, q1, q2, L1, L2,
                    dt=-dt, n_steps=1, K_ext=K_ext, r0_ext=r0_ext, mass=mass,
                    create_graph=False, record_all=False,
                )
            dx_p, dv_p, q1_p, q2_p, L1_p, L2_p = [
                prev[k].detach().requires_grad_(True) for k in ("dx", "dv", "q1", "q2", "L1", "L2")
            ]
            # Re-run 1 forward step WITH graph and compute vector-Jacobian product
            with torch.enable_grad():
                fwd = rollout_rigid_body(
                    pot_fn, dx_p, dv_p, q1_p, q2_p, L1_p, L2_p,
                    dt=dt, n_steps=1, K_ext=K_ext, r0_ext=r0_ext, mass=mass,
                    create_graph=True, record_all=False,
                )
                outs = (fwd["dx"], fwd["dv"], fwd["q1"], fwd["q2"], fwd["L1"], fwd["L2"])
                grads = torch.autograd.grad(
                    outs, (dx_p, dv_p, q1_p, q2_p, L1_p, L2_p, *params),
                    grad_outputs=adjs, allow_unused=True,
                )
            adjs = tuple(g if g is not None else torch.zeros_like(a) for g, a in zip(grads[:6], adjs))
            for i, gp in enumerate(grads[6:]):
                if gp is not None:
                    grad_params[i] = grad_params[i] + gp
            dx, dv, q1, q2, L1, L2 = dx_p.detach(), dv_p.detach(), q1_p.detach(), q2_p.detach(), L1_p.detach(), L2_p.detach()

        return (*adjs, None, None, None, None, None, None, *grad_params)


def verify_d6_symmetry() -> None:
    model = D6SymmetricCGNet(hidden=64)
    model.eval()
    torch.manual_seed(0)
    dx = torch.randn(16, 3)
    q1 = F.normalize(torch.randn(16, 4), dim=-1)
    q2 = F.normalize(torch.randn(16, 4), dim=-1)
    v0 = model(dx, q1, q2)

    # 1. Quaternion sign flip q -> -q
    v_flip = model(dx, -q1, q2)
    # 2. Exchange 1 <-> 2
    v_swap = model(-dx, q2, q1)
    # 3. 60-degree body-z rotation of hex 1: q_rot = [cos(pi/6), 0, 0, sin(pi/6)]
    c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
    def qmul(a, b):
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ], dim=-1)
    q60 = torch.tensor([[c, 0.0, 0.0, s]]).expand_as(q1)
    v_d6 = model(dx, qmul(q1, q60), q2)
    # 4. Global SO(3) rotation of entire system
    qg = F.normalize(torch.randn(16, 4), dim=-1)
    Rg = quat_to_rotmat(qg)
    dx_g = torch.einsum("bij,bj->bi", Rg, dx)
    v_so3 = model(dx_g, qmul(qg, q1), qmul(qg, q2))

    print(
        f"D6+SO(3) symmetry check | "
        f"q->-q: {(v0-v_flip).abs().max():.1e} | "
        f"1<->2: {(v0-v_swap).abs().max():.1e} | "
        f"60deg: {(v0-v_d6).abs().max():.1e} | "
        f"SO(3): {(v0-v_so3).abs().max():.1e}"
    )


def train_static_energy(splits: dict, epochs: int = 2000, batch_size: int = 512, seed: int = 42) -> tuple[nn.Module, float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 10)
    model = D6SymmetricCGNet(hidden=96)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    t0 = time.time()
    for ep in range(epochs):
        b = sample_batch(splits["train"], batch_size, n_steps=1, stratified=True, gen=gen)
        dx, q1, q2 = b["dx"][:, 0], b["q1"][:, 0], b["q2"][:, 0]
        with torch.no_grad():
            v_true = exact_hex_potential(dx, q1, q2)
        opt.zero_grad(set_to_none=True)
        v_pred = model(dx, q1, q2)
        # Huber / log-weighted MSE handles both small energies and steep 50+ walls cleanly
        loss = F.smooth_l1_loss(v_pred, v_true, beta=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if (ep + 1) % 500 == 0:
            print(f"  [static_energy] ep {ep+1:4d}/{epochs} | loss={loss.item():.5f} ({time.time()-t0:.1f}s)")
    return model, time.time() - t0


def train_static_force(splits: dict, epochs: int = 2000, batch_size: int = 512, seed: int = 42) -> tuple[nn.Module, float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 20)
    model = D6SymmetricCGNet(hidden=96)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    t0 = time.time()
    for ep in range(epochs):
        b = sample_batch(splits["train"], batch_size, n_steps=1, stratified=True, gen=gen)
        dx, q1, q2 = b["dx"][:, 0], b["q1"][:, 0], b["q2"][:, 0]
        # Ground truth force & torques (no graph)
        F2_t, tau1_t, tau2_t = compute_forces_and_torques(exact_hex_potential, dx, q1, q2, 0.0, 2.0, create_graph=False)
        opt.zero_grad(set_to_none=True)
        F2_p, tau1_p, tau2_p = compute_forces_and_torques(model, dx, q1, q2, 0.0, 2.0, create_graph=True)
        loss = (
            F.smooth_l1_loss(F2_p, F2_t, beta=2.0)
            + 0.5 * F.smooth_l1_loss(tau1_p, tau1_t, beta=2.0)
            + 0.5 * F.smooth_l1_loss(tau2_p, tau2_t, beta=2.0)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if (ep + 1) % 500 == 0:
            print(f"  [static_force ] ep {ep+1:4d}/{epochs} | loss={loss.item():.5f} ({time.time()-t0:.1f}s)")
    return model, time.time() - t0


def train_neural_ode(splits: dict, epochs: int = 1600, batch_size: int = 512, seed: int = 42) -> tuple[nn.Module, float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 30)
    meta = splits["meta"]
    scales = splits["scales"]
    model = D6SymmetricCGNet(hidden=96)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    t0 = time.time()
    for ep in range(epochs):
        n_steps = 8 if ep < 600 else (15 if ep < 1100 else 25)
        b = sample_batch(splits["train"], batch_size, n_steps=n_steps, stratified=True, gen=gen)
        opt.zero_grad(set_to_none=True)
        pred = rollout_rigid_body(
            model,
            b["dx"][:, 0], b["dv"][:, 0],
            b["q1"][:, 0], b["q2"][:, 0],
            b["L1"][:, 0], b["L2"][:, 0],
            dt=meta["dt"], n_steps=n_steps,
            K_ext=meta["K_ext"], r0_ext=meta["r0_ext"], mass=meta["mass"],
            create_graph=True, record_all=True,
        )
        l_dx = ((pred["dx"] - b["dx"]) / scales["dx"]).pow(2).mean()
        l_dv = ((pred["dv"] - b["dv"]) / scales["dv"]).pow(2).mean()
        l_L = 0.5 * (
            ((pred["L1"] - b["L1"]) / scales["L"]).pow(2).mean()
            + ((pred["L2"] - b["L2"]) / scales["L"]).pow(2).mean()
        )
        q1_dot = (pred["q1"] * b["q1"]).sum(dim=-1).clamp(-1.0, 1.0)
        q2_dot = (pred["q2"] * b["q2"]).sum(dim=-1).clamp(-1.0, 1.0)
        l_q = 0.5 * ((1.0 - q1_dot.pow(2)).mean() + (1.0 - q2_dot.pow(2)).mean()) / scales["q_chord"]
        loss = l_dx + l_dv + l_L + 0.5 * l_q

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
        if (ep + 1) % 400 == 0:
            print(f"  [neural_ode   ] ep {ep+1:4d}/{epochs} | T={n_steps:2d} | loss={loss.item():.5f} ({time.time()-t0:.1f}s)")
    return model, time.time() - t0


def evaluate_paradigm(model: nn.Module, splits: dict, label: str) -> dict:
    model.eval()
    meta = splits["meta"]
    te = splits["test"]
    dx = te["dx"][0]
    q1, q2 = te["q1"][0], te["q2"][0]
    v_ex = exact_hex_potential(dx, q1, q2)
    coll = torch.nonzero(v_ex > 0.05, as_tuple=False).squeeze(-1)
    idx = coll[:: max(1, len(coll) // 2000)]

    with torch.no_grad():
        ve = v_ex[idx]
        vp = model(dx[idx], q1[idx], q2[idx])
        v_mae = float((ve - vp).abs().mean().item())
        v_corr = float(np.corrcoef(ve.numpy(), vp.numpy())[0, 1])

    idx5 = idx[:500]
    F2_e, t1_e, _ = compute_forces_and_torques(exact_hex_potential, dx[idx5], q1[idx5], q2[idx5], 0.0, 2.0, create_graph=False)
    F2_p, t1_p, _ = compute_forces_and_torques(model, dx[idx5], q1[idx5], q2[idx5], 0.0, 2.0, create_graph=False)
    f_mae = float((F2_e - F2_p).abs().mean().item())
    t_mae = float((t1_e - t1_p).abs().mean().item())
    f_cos = float(F.cosine_similarity(F2_e, F2_p).mean().item())
    t_cos = float(F.cosine_similarity(t1_e, t1_p).mean().item())

    # 1,500-step held-out test collision trajectory error
    r_all = torch.linalg.norm(dx, dim=-1)
    cand = torch.nonzero(r_all[1000:50000] < 2.1, as_tuple=False)
    t_peak = int(cand[0].item()) + 1000
    t0 = max(0, t_peak - 500)
    n_steps = 1500
    with torch.enable_grad():
        pred = rollout_rigid_body(
            model, te["dx"][0:1, t0], te["dv"][0:1, t0],
            te["q1"][0:1, t0], te["q2"][0:1, t0],
            te["L1"][0:1, t0], te["L2"][0:1, t0],
            dt=meta["dt"], n_steps=n_steps, K_ext=meta["K_ext"], r0_ext=meta["r0_ext"], mass=meta["mass"],
            create_graph=False, record_all=True,
        )
    true_dx = te["dx"][0, t0 : t0 + n_steps + 1]
    true_L1 = te["L1"][0, t0 : t0 + n_steps + 1]
    true_q1 = te["q1"][0, t0 : t0 + n_steps + 1]
    traj_r_mae = float((torch.linalg.norm(pred["dx"][0], dim=-1) - torch.linalg.norm(true_dx, dim=-1)).abs().mean().item())
    traj_L_mae = float((torch.linalg.norm(pred["L1"][0], dim=-1) - torch.linalg.norm(true_L1, dim=-1)).abs().mean().item())
    dot = (pred["q1"][0] * true_q1).sum(dim=-1).abs().clamp(0.0, 1.0)
    traj_deg_final = float((2.0 * torch.acos(dot[-1]) * (180.0 / math.pi)).item())

    res = {
        "v_mae_coll": v_mae,
        "v_corr_coll": v_corr,
        "f_mae": f_mae,
        "f_cos": f_cos,
        "t_mae": t_mae,
        "t_cos": t_cos,
        "traj_r_mae": traj_r_mae,
        "traj_L_mae": traj_L_mae,
        "traj_deg_final": traj_deg_final,
    }
    print(
        f"  Eval [{label:13s}] V_MAE={v_mae:.3f}, V_corr={v_corr:.4f} | "
        f"F_MAE={f_mae:.2f} (cos={f_cos:.3f}), T_MAE={t_mae:.2f} (cos={t_cos:.3f}) | "
        f"1500-step r_MAE={traj_r_mae:.4f}, L_MAE={traj_L_mae:.3f}, end_deg={traj_deg_final:.1f}°"
    )
    return res, pred


def benchmark_speed_and_memory(splits: dict) -> dict:
    """Measure step time (ms) and peak Python/Torch allocation (MB) vs horizon T
    for Static Energy, Static Force, Neural ODE (BPTT), and Neural ODE (Symplectic Adjoint).
    """
    print("\n=== Benchmarking Speed & Peak Memory Scaling vs Horizon T ===")
    meta = splits["meta"]
    gen = torch.Generator().manual_seed(777)
    batch_size = 256
    horizons = [10, 20, 40, 80]

    def measure(fn):
        gc.collect()
        # Warmup
        fn()
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return dt_ms, peak_bytes / (1024 * 1024)

    # Also measure real RSS / PyTorch tensor bytes using torch.profiler for accurate C++ tensor memory!
    def measure_torch(fn):
        gc.collect()
        fn()  # warmup
        gc.collect()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
        ) as prof:
            t0 = time.perf_counter()
            fn()
            dt_ms = (time.perf_counter() - t0) * 1000.0
        # Sum peak self_cpu_memory_usage (positive allocations) as proxy for graph memory
        alloc_mb = sum(max(0, e.cpu_memory_usage) for e in prof.key_averages()) / (1024 * 1024)
        return dt_ms, alloc_mb

    model = D6SymmetricCGNet(hidden=96)
    b = sample_batch(splits["train"], batch_size, n_steps=80, stratified=True, gen=gen)
    dx0, dv0, q1_0, q2_0, L1_0, L2_0 = b["dx"][:, 0], b["dv"][:, 0], b["q1"][:, 0], b["q2"][:, 0], b["L1"][:, 0], b["L2"][:, 0]
    v_t = exact_hex_potential(dx0, q1_0, q2_0).detach()
    F2_t, t1_t, t2_t = compute_forces_and_torques(exact_hex_potential, dx0, q1_0, q2_0, 0.0, 2.0, create_graph=False)
    F2_t, t1_t, t2_t = F2_t.detach(), t1_t.detach(), t2_t.detach()

    def step_static_energy():
        model.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(dx0, q1_0, q2_0), v_t)
        loss.backward()

    def step_static_force():
        model.zero_grad(set_to_none=True)
        F2_p, t1_p, t2_p = compute_forces_and_torques(model, dx0, q1_0, q2_0, 0.0, 2.0, create_graph=True)
        loss = (F2_p - F2_t).pow(2).mean() + (t1_p - t1_t).pow(2).mean() + (t2_p - t2_t).pow(2).mean()
        loss.backward()

    se_ms, se_mb = measure_torch(step_static_energy)
    sf_ms, sf_mb = measure_torch(step_static_force)
    print(f"  Static Energy: {se_ms:6.1f} ms/step | {se_mb:6.1f} MB graph alloc")
    print(f"  Static Force : {sf_ms:6.1f} ms/step | {sf_mb:6.1f} MB graph alloc")

    bptt_ms, bptt_mb = [], []
    adj_ms, adj_mb = [], []
    for T in horizons:
        def step_bptt():
            model.zero_grad(set_to_none=True)
            p = rollout_rigid_body(
                model, dx0, dv0, q1_0, q2_0, L1_0, L2_0,
                dt=meta["dt"], n_steps=T, K_ext=meta["K_ext"], r0_ext=meta["r0_ext"], mass=meta["mass"],
                create_graph=True, record_all=False,
            )
            loss = (p["dx"] - b["dx"][:, T]).pow(2).mean() + (p["dv"] - b["dv"][:, T]).pow(2).mean()
            loss.backward()

        def step_adjoint():
            model.zero_grad(set_to_none=True)
            dx_f, dv_f, q1_f, q2_f, L1_f, L2_f = ReversibleVerletStep.apply(
                dx0, dv0, q1_0, q2_0, L1_0, L2_0,
                meta["dt"], T, meta["K_ext"], meta["r0_ext"], meta["mass"], model,
                *list(model.parameters()),
            )
            loss = (dx_f - b["dx"][:, T]).pow(2).mean() + (dv_f - b["dv"][:, T]).pow(2).mean()
            loss.backward()

        t_b, m_b = measure_torch(step_bptt)
        t_a, m_a = measure_torch(step_adjoint)
        bptt_ms.append(t_b); bptt_mb.append(m_b)
        adj_ms.append(t_a); adj_mb.append(m_a)
        print(f"  Horizon T={T:2d} | BPTT: {t_b:6.1f} ms, {m_b:6.1f} MB | Adjoint: {t_a:6.1f} ms, {m_a:6.1f} MB")

    return {
        "batch_size": batch_size,
        "horizons": horizons,
        "static_energy": {"ms": se_ms, "mb": se_mb},
        "static_force": {"ms": sf_ms, "mb": sf_mb},
        "bptt": {"ms": bptt_ms, "mb": bptt_mb},
        "adjoint": {"ms": adj_ms, "mb": adj_mb},
    }


def plot_paradigm_comparison(preds: dict, splits: dict, bench: dict, out_traj: Path, out_cost: Path) -> None:
    set_style()
    meta = splits["meta"]
    te = splits["test"]
    dx = te["dx"][0]
    r_all = torch.linalg.norm(dx, dim=-1)
    cand = torch.nonzero(r_all[1000:50000] < 2.1, as_tuple=False)
    t_peak = int(cand[0].item()) + 1000
    t0 = max(0, t_peak - 500)
    n_steps = 1500
    t_axis = np.arange(n_steps + 1) * meta["dt"]

    true_dx = te["dx"][0, t0 : t0 + n_steps + 1]
    true_dv = te["dv"][0, t0 : t0 + n_steps + 1]
    true_L1 = te["L1"][0, t0 : t0 + n_steps + 1]
    true_q1 = te["q1"][0, t0 : t0 + n_steps + 1]

    colors = {"static_energy": "tab:green", "static_force": "tab:purple", "neural_ode": "tab:blue"}
    labels = {
        "static_energy": "Static Energy-Matching (needs V labels)",
        "static_force": "Static Force+Torque (needs F, τ labels)",
        "neural_ode": "Neural ODE (trajectories only!)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.4), sharex=True)
    ax = axes[0, 0]
    ax.plot(t_axis, torch.linalg.norm(true_dx, dim=-1).numpy(), "--", color="black", lw=1.8, label="Ground truth (LAMMPS)")
    for k in ("static_energy", "static_force", "neural_ode"):
        ax.plot(t_axis, torch.linalg.norm(preds[k]["dx"][0].detach(), dim=-1).numpy(), color=colors[k], lw=1.4, label=labels[k], alpha=0.9)
    ax.set_ylabel(r"$\|\mathbf{x}_2 - \mathbf{x}_1\|\ /\ \sigma_{\mathrm{unit}}$")
    ax.set_title("a) Relative Separation (Held-Out Test Collision)")
    ax.legend(loc="best", frameon=True, fontsize=7.5)

    ax = axes[0, 1]
    ax.plot(t_axis, torch.linalg.norm(true_dv, dim=-1).numpy(), "--", color="black", lw=1.8)
    for k in ("static_energy", "static_force", "neural_ode"):
        ax.plot(t_axis, torch.linalg.norm(preds[k]["dv"][0].detach(), dim=-1).numpy(), color=colors[k], lw=1.4, alpha=0.9)
    ax.set_ylabel(r"$\|\mathbf{v}_2 - \mathbf{v}_1\|$")
    ax.set_title("b) Relative Speed")

    ax = axes[1, 0]
    ax.plot(t_axis, torch.linalg.norm(true_L1, dim=-1).numpy(), "--", color="black", lw=1.8)
    for k in ("static_energy", "static_force", "neural_ode"):
        ax.plot(t_axis, torch.linalg.norm(preds[k]["L1"][0].detach(), dim=-1).numpy(), color=colors[k], lw=1.4, alpha=0.9)
    ax.set_xlabel(r"Time / $\tau_{\mathrm{unit}}$")
    ax.set_ylabel(r"$\|\mathbf{L}_1\|$")
    ax.set_title("c) Body 1 Angular Momentum Magnitude")

    ax = axes[1, 1]
    for k in ("static_energy", "static_force", "neural_ode"):
        q_p = preds[k]["q1"][0].detach()
        dot = (q_p * true_q1).sum(dim=-1).abs().clamp(0.0, 1.0)
        deg = (2.0 * torch.acos(dot) * (180.0 / np.pi)).numpy()
        ax.plot(t_axis, deg, color=colors[k], lw=1.4, alpha=0.9)
    ax.set_xlabel(r"Time / $\tau_{\mathrm{unit}}$")
    ax.set_ylabel("Orientation Error (deg)")
    ax.set_title("d) Body 1 Orientation Error vs. LAMMPS")

    fig.suptitle("D6 + SO(3) + C² Cutoff CGNet: Static Point-Energy / Force Matching vs. Neural ODE", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_traj)
    plt.close(fig)

    # Cost & Memory Scaling Plot
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    H = bench["horizons"]

    ax = axes[0]
    ax.plot(H, bench["bptt"]["ms"], "-o", color="tab:red", lw=1.8, label="Neural ODE (Direct Backprop / BPTT)")
    ax.plot(H, bench["adjoint"]["ms"], "-s", color="tab:blue", lw=1.8, label="Neural ODE (Symplectic Adjoint)")
    ax.axhline(bench["static_force"]["ms"], ls="--", color="tab:purple", lw=1.4, label=f"Static Force+Torque ({bench['static_force']['ms']:.1f} ms)")
    ax.axhline(bench["static_energy"]["ms"], ls=":", color="tab:green", lw=1.4, label=f"Static Energy ({bench['static_energy']['ms']:.1f} ms)")
    ax.set_xlabel("Integration Horizon T (Velocity-Verlet steps)")
    ax.set_ylabel("Wall-Clock Time per Step (ms)")
    ax.set_title("a) Compute Time Scaling (Speed Trade-off)")
    ax.legend(loc="upper left", frameon=True, fontsize=7.8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(H, bench["bptt"]["mb"], "-o", color="tab:red", lw=1.8, label="Neural ODE (BPTT — O(T) memory)")
    ax.plot(H, bench["adjoint"]["mb"], "-s", color="tab:blue", lw=1.8, label="Neural ODE (Adjoint — O(1) memory)")
    ax.axhline(bench["static_force"]["mb"], ls="--", color="tab:purple", lw=1.4, label=f"Static Force+Torque ({bench['static_force']['mb']:.1f} MB)")
    ax.axhline(bench["static_energy"]["mb"], ls=":", color="tab:green", lw=1.4, label=f"Static Energy ({bench['static_energy']['mb']:.1f} MB)")
    ax.set_xlabel("Integration Horizon T (Velocity-Verlet steps)")
    ax.set_ylabel("Autograd Graph Allocation (MB)")
    ax.set_title("b) Memory Cost Scaling (Adjoint vs. BPTT)")
    ax.legend(loc="upper left", frameon=True, fontsize=7.8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Computational Cost & Memory Scaling (Batch Size = {bench['batch_size']}, D6+SO(3) CGNet)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_cost)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    t_all = time.time()
    verify_d6_symmetry()

    d = torch.load(DATA_PATH, weights_only=True)
    splits = prepare_splits(d)

    print("\n=== 1. Training Static Energy-Matching ===")
    m_energy, t_energy = train_static_energy(splits, epochs=2000, batch_size=512)
    print("\n=== 2. Training Static Force+Torque Matching ===")
    m_force, t_force = train_static_force(splits, epochs=2000, batch_size=512)
    print("\n=== 3. Training Neural ODE (Trajectories Only) ===")
    m_ode, t_ode = train_neural_ode(splits, epochs=1600, batch_size=512)

    print("\n=== Held-Out Test Evaluation ===")
    res_e, pred_e = evaluate_paradigm(m_energy, splits, "static_energy")
    res_f, pred_f = evaluate_paradigm(m_force, splits, "static_force")
    res_o, pred_o = evaluate_paradigm(m_ode, splits, "neural_ode")

    res_e["train_time_s"] = t_energy
    res_f["train_time_s"] = t_force
    res_o["train_time_s"] = t_ode

    bench = benchmark_speed_and_memory(splits)

    plot_paradigm_comparison(
        {"static_energy": pred_e, "static_force": pred_f, "neural_ode": pred_o},
        splits, bench,
        FIG_DIR / "cg_paradigm_trajectories.png",
        FIG_DIR / "cg_paradigm_cost_memory.png",
    )

    with open(OUT_DIR / "paradigm_comparison.json", "w") as f:
        json.dump({"static_energy": res_e, "static_force": res_f, "neural_ode": res_o, "benchmark": bench}, f, indent=2)

    print(f"\nAll done in {time.time() - t_all:.1f}s!")


if __name__ == "__main__":
    main()
