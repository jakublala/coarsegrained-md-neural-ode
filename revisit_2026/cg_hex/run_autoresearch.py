"""Autoresearch hill-climbing loop to recover the exact atomistic rigid-body trajectory
purely from COM + quaternion trajectories (Neural ODE training, NO energy/force labels).

Tracks a single scalar metric D (Normalized Trajectory Deviation over 10 held-out test collisions
x 200 steps; D=1.0 is V_CG=0, D=0.0 is exact LAMMPS) and plots the full hill-climbing progress
on a single figure: figures/autoresearch_progress.png (plus winning trajectory overlay).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rigid_ode import (
    RawMLPNet,
    ImprovedCGNet,
    quat_to_rotmat,
    smooth_cutoff,
    exact_hex_potential,
    rollout_rigid_body,
    R_MAX,
    R_ON,
)
from compare_paradigms import D6SymmetricCGNet, _cos6_smooth
from train_compare import prepare_splits, sample_batch, set_style
from autoresearch_bench import build_benchmark, score_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hex_dataset.pt"
OUT_DIR = ROOT / "results"
FIG_DIR = ROOT.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# --- Candidate Architectures for Autoresearch ---

class ResD6TwistCGNet(nn.Module):
    """SO(3) + D6 + relative in-plane twist cos(6*psi) + RBF radial encoding + Residual MLP."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.register_buffer("rbf_centers", torch.linspace(1.0, R_MAX, 12))
        # 12 RBF(r) + 12 D6 features + 1 relative twist = 25
        self.in_proj = nn.Linear(25, hidden)
        self.b1 = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.b2 = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.out_proj = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def _pair_features(self, r, u12, u21, R12):
        rbf = torch.exp(-((r.unsqueeze(-1) - self.rbf_centers) / 0.25) ** 2)
        rho1 = torch.sqrt(u12[:, 0] ** 2 + u12[:, 1] ** 2 + 1e-8)
        z1_sq = u12[:, 2] ** 2
        c6_1 = _cos6_smooth(u12[:, 0], u12[:, 1])
        rho2 = torch.sqrt(u21[:, 0] ** 2 + u21[:, 1] ** 2 + 1e-8)
        z2_sq = u21[:, 2] ** 2
        c6_2 = _cos6_smooth(u21[:, 0], u21[:, 1])
        n2_in_1 = R12[:, :, 2]
        n1_n2_sq = n2_in_1[:, 2] ** 2
        rho_n2 = torch.sqrt(n2_in_1[:, 0] ** 2 + n2_in_1[:, 1] ** 2 + 1e-8)
        c6_n2 = _cos6_smooth(n2_in_1[:, 0], n2_in_1[:, 1])
        proj_sq = (u12[:, 0] * n2_in_1[:, 0] + u12[:, 1] * n2_in_1[:, 1]) ** 2
        # Relative in-plane twist angle psi between x1 and x2 axes:
        # cos(psi) ~ R12[0,0] + R12[1,1], sin(psi) ~ R12[1,0] - R12[0,1] (for parallel planes)
        c_psi = R12[:, 0, 0] + R12[:, 1, 1]
        s_psi = R12[:, 1, 0] - R12[:, 0, 1]
        twist6 = _cos6_smooth(c_psi, s_psi)
        base = torch.stack([
            r, 1.0 / (r + 0.1), rho1, z1_sq, c6_1,
            rho2, z2_sq, c6_2, n1_n2_sq, rho_n2, c6_n2, proj_sq, twist6,
        ], dim=-1)
        return torch.cat([rbf, base], dim=-1)

    def forward(self, dx, q1, q2):
        R1 = quat_to_rotmat(q1)
        R2 = quat_to_rotmat(q2)
        r = torch.linalg.norm(dx, dim=-1).clamp(min=1e-6)
        u12 = torch.einsum("bij,bi->bj", R1, dx)
        u21 = torch.einsum("bij,bi->bj", R2, -dx)
        R12 = torch.matmul(R1.transpose(-1, -2), R2)
        R21 = R12.transpose(-1, -2)

        def eval_one(f):
            h = self.in_proj(f)
            h = h + self.b1(h)
            h = h + self.b2(h)
            return self.out_proj(h).squeeze(-1)

        raw = 0.5 * (eval_one(self._pair_features(r, u12, u21, R12)) + eval_one(self._pair_features(r, u21, u12, R21)))
        env = smooth_cutoff(r, R_ON, R_MAX)
        return env * (F.softplus(raw) ** 2.5)


class LatentSiteCGNet(nn.Module):
    """General physics-informed Neural CG Potential for ANY composite rigid body:
    Learns K effective 3D interaction sites {s_k} in the body frame AND a smooth
    radial neural pair potential phi(d) purely from COM + quaternion trajectories!

    Strictly SO(3)-invariant, q == -q invariant, 1 <-> 2 exchange symmetric,
    C^2 cutoff-enveloped, and naturally represents arbitrarily sharp 3D collisions.
    """

    def __init__(self, n_sites: int = 7, hidden: int = 48, r_cut_site: float = 1.35, init_scale: float = 0.85, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # Initialize K learnable 3D site coordinates randomly in [-init_scale, init_scale] (NO hexagon prior!)
        sites_init = torch.randn(n_sites, 3, generator=gen) * init_scale
        self.sites = nn.Parameter(sites_init)
        self.r_cut_site = r_cut_site
        self.r_on_site = 0.85

        # Smooth neural radial potential phi(d) taking [d, 1/d, 1/d^2, RBF(d)]
        self.register_buffer("rbf_c", torch.linspace(0.7, r_cut_site, 8))
        self.rad_net = nn.Sequential(
            nn.Linear(11, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.rad_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, dx: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        # Center sites around body COM (mean zero)
        s = self.sites - self.sites.mean(dim=0, keepdim=True)  # (K, 3)
        R1 = quat_to_rotmat(q1)  # (B, 3, 3)
        R2 = quat_to_rotmat(q2)  # (B, 3, 3)
        p1 = torch.einsum("bij,kj->bki", R1, s)                     # (B, K, 3)
        p2 = torch.einsum("bij,kj->bki", R2, s) + dx[:, None, :]    # (B, K, 3)
        diff = p2[:, None, :, :] - p1[:, :, None, :]                # (B, K, K, 3)
        d = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6)              # (B, K, K)

        # C^2 smooth cutoff on site-site distance d
        env = smooth_cutoff(d, self.r_on_site, self.r_cut_site)
        inv_d = 1.0 / d.clamp(min=0.35)
        rbf = torch.exp(-((d.unsqueeze(-1) - self.rbf_c) / 0.15) ** 2)
        feats = torch.cat([d.unsqueeze(-1), inv_d.unsqueeze(-1), (inv_d ** 2).unsqueeze(-1), rbf], dim=-1)
        raw = self.rad_net(feats).squeeze(-1)
        v_pair = env * (F.softplus(raw) ** 2)
        return v_pair.sum(dim=(-1, -2))


def sample_impulse_weighted_batch(split: dict, weights: torch.Tensor, batch_size: int, n_steps: int, gen: torch.Generator) -> dict:
    """Sample 80% from impulse-weighted collision frames (emphasizing hard impacts) + 20% near-contact."""
    max_t = split["dx"].shape[1] - n_steps - 1
    n_act = int(0.8 * batch_size)
    n_near = batch_size - n_act

    sel = torch.multinomial(weights, n_act, replacement=True, generator=gen)
    jitter = torch.randint(0, 6, (n_act,), generator=gen)
    ti_act = split["active_idx"][sel, 0]
    t0_act = (split["active_idx"][sel, 1] - jitter).clamp(min=0, max=max_t)

    sel_n = torch.randint(0, len(split["near_idx"]), (n_near,), generator=gen)
    ti_near = split["near_idx"][sel_n, 0]
    t0_near = split["near_idx"][sel_n, 1].clamp(max=max_t)

    ti = torch.cat([ti_act, ti_near], dim=0)
    t0 = torch.cat([t0_act, t0_near], dim=0)
    offsets = torch.arange(n_steps + 1).unsqueeze(0)
    t_mat = t0.unsqueeze(1) + offsets
    ti_mat = ti.unsqueeze(1)
    return {
        "dx": split["dx"][ti_mat, t_mat],
        "dv": split["dv"][ti_mat, t_mat],
        "q1": split["q1"][ti_mat, t_mat],
        "q2": split["q2"][ti_mat, t_mat],
        "L1": split["L1"][ti_mat, t_mat],
        "L2": split["L2"][ti_mat, t_mat],
    }


def train_ode_candidate(
    model: nn.Module,
    splits: dict,
    imp_weights: torch.Tensor | None,
    epochs: int,
    lr: float,
    batch_size: int = 512,
    horizons: tuple[int, int, int] = (8, 16, 25),
    use_huber: bool = False,
    seed: int = 42,
    tag: str = "",
) -> nn.Module:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 500)
    meta = splits["meta"]
    scales = splits["scales"]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)

    t0 = time.time()
    for ep in range(epochs):
        n_steps = horizons[0] if ep < epochs // 3 else (horizons[1] if ep < 2 * epochs // 3 else horizons[2])
        if imp_weights is not None:
            b = sample_impulse_weighted_batch(splits["train"], imp_weights, batch_size, n_steps, gen)
        else:
            b = sample_batch(splits["train"], batch_size, n_steps, stratified=True, gen=gen)

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

        err_dx = (pred["dx"] - b["dx"]) / scales["dx"]
        err_dv = (pred["dv"] - b["dv"]) / scales["dv"]
        err_L1 = (pred["L1"] - b["L1"]) / scales["L"]
        err_L2 = (pred["L2"] - b["L2"]) / scales["L"]

        if use_huber:
            l_dx = F.smooth_l1_loss(err_dx, torch.zeros_like(err_dx), beta=1.0)
            l_dv = F.smooth_l1_loss(err_dv, torch.zeros_like(err_dv), beta=1.0)
            l_L = 0.5 * (
                F.smooth_l1_loss(err_L1, torch.zeros_like(err_L1), beta=1.0)
                + F.smooth_l1_loss(err_L2, torch.zeros_like(err_L2), beta=1.0)
            )
        else:
            l_dx = err_dx.pow(2).mean()
            l_dv = err_dv.pow(2).mean()
            l_L = 0.5 * (err_L1.pow(2).mean() + err_L2.pow(2).mean())

        q1_dot = (pred["q1"] * b["q1"]).sum(dim=-1).clamp(-1.0, 1.0)
        q2_dot = (pred["q2"] * b["q2"]).sum(dim=-1).clamp(-1.0, 1.0)
        l_q = 0.5 * ((1.0 - q1_dot.pow(2)).mean() + (1.0 - q2_dot.pow(2)).mean()) / scales["q_chord"]
        loss = l_dx + 2.0 * l_dv + 2.0 * l_L + 0.5 * l_q

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 15.0)
            opt.step()
        sched.step()
        if (ep + 1) % max(1, epochs // 3) == 0:
            print(f"    [{tag}] ep {ep+1:4d}/{epochs} | T={n_steps:2d} | loss={loss.item():.5f} ({time.time()-t0:.1f}s)")
    return model


def plot_autoresearch_progress(history: list[dict], best_model: nn.Module, bench: dict, out_path: Path) -> None:
    """Single clean Autoresearch progress figure:
    Left:  Hill-climbing trajectory deviation D vs Iteration # (kept frontier + discarded trials).
    Right: Winning Neural ODE CG model vs Exact LAMMPS on a held-out test collision.
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # --- Left: Autoresearch Progress Plot (Single Metric D) ---
    ax = axes[0]
    iters = [h["iter"] for h in history]
    Ds = [h["D"] for h in history]
    kept = [h["kept"] for h in history]

    # Frontier step line (running best)
    best_so_far = []
    cur = float("inf")
    for d_val in Ds:
        cur = min(cur, d_val)
        best_so_far.append(cur)

    ax.step(iters, best_so_far, where="post", color="#16a34a", lw=2.2, label="Running Best Frontier", zorder=3)

    # Plot kept vs discarded points
    for h in history:
        if h["kept"]:
            ax.scatter(h["iter"], h["D"], color="#16a34a", s=65, zorder=5, edgecolors="black", linewidths=0.7)
            offset_y = 1.22 if h["D"] > 0.05 else 1.30
            ax.annotate(
                f"{h['short']}\n(D={h['D']:.4f})",
                (h["iter"], h["D"]),
                xytext=(h["iter"] + 0.08, h["D"] * offset_y),
                fontsize=7.3,
                fontweight="bold",
                color="#14532d",
                arrowprops=dict(arrowstyle="-", color="#16a34a", lw=0.6, alpha=0.7),
            )
        else:
            ax.scatter(h["iter"], h["D"], color="#9ca3af", s=45, marker="x", zorder=4, linewidths=1.4)
            ax.annotate(
                f"{h['short']}\n(D={h['D']:.3f})",
                (h["iter"], h["D"]),
                xytext=(h["iter"] - 0.15, h["D"] * 1.22),
                fontsize=7.0,
                color="#6b7280",
            )

    ax.axhline(1.0, ls=":", color="tab:orange", lw=1.1, label="Trivial V_CG = 0 (D = 1.0)")
    ax.axhline(history[0]["exact_floor"], ls="--", color="black", lw=1.1, label=f"Exact 7-Bead LJ Floor (D = {history[0]['exact_floor']:.4f})")
    ax.set_yscale("log")
    ax.set_xlim(-0.4, len(history) - 0.1)
    ax.set_ylim(history[0]["exact_floor"] * 0.5, max(Ds) * 2.4)
    ax.set_xticks(iters)
    ax.set_xlabel("Autoresearch Iteration")
    ax.set_ylabel("Held-Out Trajectory Deviation Metric D (log scale)")
    ax.set_title("a) Autoresearch Hill-Climb: Single Trajectory Deviation Metric D")
    ax.legend(loc="lower left", frameon=True, fontsize=7.6)
    ax.grid(alpha=0.25, which="both")

    # --- Right: Best Model vs LAMMPS on Held-Out Test Collision #0 ---
    ax = axes[1]
    best_model.eval()
    with torch.enable_grad():
        p = rollout_rigid_body(
            best_model,
            bench["dx"][0:1, 0], bench["dv"][0:1, 0],
            bench["q1"][0:1, 0], bench["q2"][0:1, 0],
            bench["L1"][0:1, 0], bench["L2"][0:1, 0],
            dt=bench["dt"], n_steps=bench["n_steps"],
            K_ext=bench["K_ext"], r0_ext=bench["r0_ext"], mass=bench["mass"],
            create_graph=False, record_all=True,
        )
    t_ax = np.arange(bench["n_steps"] + 1) * bench["dt"]
    r_true = torch.linalg.norm(bench["dx"][0], dim=-1).numpy()
    r_pred = torch.linalg.norm(p["dx"][0].detach(), dim=-1).numpy()
    L_true = torch.linalg.norm(bench["L1"][0], dim=-1).numpy()
    L_pred = torch.linalg.norm(p["L1"][0].detach(), dim=-1).numpy()

    ax2 = ax.twinx()
    l1, = ax.plot(t_ax, r_true, "--", color="black", lw=2.0, label="LAMMPS separation r(t)")
    l2, = ax.plot(t_ax, r_pred, "-", color="#2563eb", lw=1.8, label="Best Neural ODE r(t)")
    l3, = ax2.plot(t_ax, L_true, ":", color="#dc2626", lw=2.0, label="LAMMPS ||L1(t)||")
    l4, = ax2.plot(t_ax, L_pred, "-", color="#f97316", lw=1.6, alpha=0.85, label="Best Neural ODE ||L1(t)||")

    ax.set_xlabel(r"Time / $\tau_{\mathrm{unit}}$")
    ax.set_ylabel(r"Separation $\|\Delta\mathbf{x}\|$ / $\sigma_{\mathrm{unit}}$", color="#2563eb")
    ax2.set_ylabel(r"Body 1 Angular Momentum $\|\mathbf{L}_1\|$", color="#dc2626")
    ax.set_title("b) Best Neural ODE vs. Atomistic LAMMPS (Test Collision)")
    lines = [l1, l2, l3, l4]
    ax.legend(lines, [l.get_label() for l in lines], loc="best", frameon=True, fontsize=7.4)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    t_all = time.time()
    d = torch.load(DATA_PATH, weights_only=True)
    splits = prepare_splits(d)
    bench = build_benchmark(d, n_windows=10, n_steps=200)
    exact_floor = score_model(exact_hex_potential, bench)["D"]
    print(f"Autoresearch benchmark ready | Exact LJ floor D = {exact_floor:.6f} | Trivial V=0 D = 1.0000")

    # Compute impulse weights on training active frames (100% trajectory-only: |dL1| + |dL2|)
    tr = splits["train"]
    ti_a, t0_a = tr["active_idx"][:, 0], tr["active_idx"][:, 1]
    dL_act = (tr["L1"][ti_a, t0_a + 1] - tr["L1"][ti_a, t0_a]).norm(dim=-1) + (
        tr["L2"][ti_a, t0_a + 1] - tr["L2"][ti_a, t0_a]
    ).norm(dim=-1)
    imp_weights = (dL_act + 1e-3).pow(0.6)
    imp_weights = imp_weights / imp_weights.sum()

    history = []
    best_D = float("inf")
    best_model = None

    def record(it: int, short: str, desc: str, model: nn.Module):
        nonlocal best_D, best_model
        sc = score_model(model, bench)
        kept = sc["D"] < best_D
        if kept:
            best_D = sc["D"]
            best_model = model
            torch.save(model.state_dict(), OUT_DIR / "autoresearch_best.pt")
        status = "KEEP" if kept else "DISCARD"
        print(
            f"  --> Iter {it} [{status:7s}] {short:24s} | "
            f"D = {sc['D']:.5f} (dx={sc['dx_mae']:.5f}, dv={sc['dv_mae']:.4f}, L={sc['L_mae']:.4f}, deg={sc['deg_mae']:.2f}°)"
        )
        history.append({
            "iter": it,
            "short": short,
            "desc": desc,
            "kept": kept,
            "exact_floor": exact_floor,
            **sc,
        })

    # Iter 0: Paper Baseline (trained on aligned data)
    print("\n[Iter 0] Training Paper Baseline (raw 11D MLP, uniform sampling)...")
    from train_compare import train_variant
    m0, _ = train_variant("baseline", splits, epochs=400, batch_size=512, seed=42)
    record(0, "Paper Baseline", "Raw 11D MLP, final-state MSE", m0)

    # Iter 1: SO(3) + C2 Cutoff (trained on aligned data)
    print("\n[Iter 1] Training SO(3) + C2 Cutoff (ImprovedCGNet)...")
    m1, _ = train_variant("improved", splits, epochs=600, batch_size=512, seed=42)
    record(1, "SO(3)+Cutoff", "16D SO(3) MLP + C2 cutoff + softplus", m1)

    # Iter 2: D6 + SO(3) + Twist + RBF Residual MLP + Impulse-Weighted Sampling
    print("\n[Iter 2] Training ResD6TwistCGNet (D6 + twist + RBF + impulse-weighted)...")
    m2 = train_ode_candidate(
        ResD6TwistCGNet(hidden=96), splits, imp_weights=imp_weights,
        epochs=600, lr=3e-3, batch_size=512, horizons=(8, 14, 20), use_huber=True, seed=102, tag="Iter2-ResD6",
    )
    record(2, "ResD6+Twist+RBF", "D6 + twist + RBF ResNet + impulse sampling", m2)

    # Iter 3: Latent-Site Neural CG Potential with K=4 sites (too coarse -> expected ablation!)
    print("\n[Iter 3] Training LatentSiteCGNet (K=4 learnable 3D sites, too coarse?)...")
    m3 = train_ode_candidate(
        LatentSiteCGNet(n_sites=4, hidden=48, seed=103), splits, imp_weights=imp_weights,
        epochs=700, lr=8e-3, batch_size=512, horizons=(6, 12, 20), use_huber=False, seed=103, tag="Iter3-4sites",
    )
    record(3, "LatentSite (K=4)", "4 learnable 3D sites + radial NN", m3)

    # Iter 4: Latent-Site Neural CG Potential with K=7 sites (random init, NO hexagon prior!)
    print("\n[Iter 4] Training LatentSiteCGNet (K=7 learnable 3D sites from random init)...")
    m4 = train_ode_candidate(
        LatentSiteCGNet(n_sites=7, hidden=48, seed=104), splits, imp_weights=imp_weights,
        epochs=900, lr=1e-2, batch_size=512, horizons=(6, 12, 20), use_huber=False, seed=104, tag="Iter4-7sites",
    )
    record(4, "LatentSite (K=7)", "7 learnable 3D sites + radial NN (random init)", m4)

    # Iter 5: Fine-tune K=7 LatentSite with longer horizons (15 -> 25 -> 40) + lower LR
    print("\n[Iter 5] Fine-tuning K=7 LatentSite on longer horizons (15->25->40)...")
    m5 = LatentSiteCGNet(n_sites=7, hidden=48, seed=104)
    m5.load_state_dict(m4.state_dict())
    m5 = train_ode_candidate(
        m5, splits, imp_weights=imp_weights,
        epochs=600, lr=2e-3, batch_size=512, horizons=(15, 25, 35), use_huber=False, seed=105, tag="Iter5-FineTune",
    )
    record(5, "K=7 Long-Horizon", "Fine-tuned K=7 on T=15..35 rollouts", m5)

    # Iter 6: Test K=10 sites (does over-parameterizing sites help or hurt?)
    print("\n[Iter 6] Training LatentSiteCGNet (K=10 learnable sites)...")
    m6 = train_ode_candidate(
        LatentSiteCGNet(n_sites=10, hidden=48, seed=106), splits, imp_weights=imp_weights,
        epochs=900, lr=1e-2, batch_size=512, horizons=(6, 12, 20), use_huber=False, seed=106, tag="Iter6-10sites",
    )
    record(6, "LatentSite (K=10)", "10 learnable 3D sites + radial NN", m6)

    # Iter 7: Polish best model with low-LR precision fine-tuning
    print("\n[Iter 7] Polishing best model with low-LR precision pass...")
    if isinstance(best_model, LatentSiteCGNet):
        m7 = LatentSiteCGNet(n_sites=best_model.sites.shape[0], hidden=48, seed=107)
        m7.load_state_dict(best_model.state_dict())
        m7 = train_ode_candidate(
            m7, splits, imp_weights=imp_weights,
            epochs=500, lr=5e-4, batch_size=768, horizons=(20, 30, 40), use_huber=False, seed=107, tag="Iter7-Polish",
        )
        record(7, "Precision Polish", "Low-LR long-horizon polish on best", m7)

    # Save learned site geometry if best is LatentSiteCGNet
    if isinstance(best_model, LatentSiteCGNet):
        s = (best_model.sites - best_model.sites.mean(dim=0, keepdim=True)).detach()
        radii = torch.linalg.norm(s, dim=-1).sort().values.tolist()
        print(f"\nBest LatentSiteCGNet learned site radii from COM: {[round(r, 3) for r in radii]}")

    plot_autoresearch_progress(history, best_model, bench, FIG_DIR / "autoresearch_progress.png")
    with open(OUT_DIR / "autoresearch_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nAutoresearch complete in {time.time() - t_all:.1f}s! Best D = {best_D:.5f} (exact floor = {exact_floor:.5f})")


if __name__ == "__main__":
    main()
