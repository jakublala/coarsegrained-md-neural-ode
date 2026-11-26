"""Focused Autoresearch Hill-Climb (v2) on the aligned LAMMPS dataset.

Goal: Drive the single held-out Trajectory Deviation Metric D (10 test collisions x 200 steps;
D = 1.0 is V_CG = 0, D = 0.015 is exact atomistic LJ floor) as close to exact recovery as possible,
strictly from trajectories alone (Neural ODE, NO energy/force labels).
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
from compare_paradigms import D6SymmetricCGNet
from run_autoresearch import (
    ResD6TwistCGNet,
    LatentSiteCGNet,
    train_ode_candidate,
    plot_autoresearch_progress,
)
from train_compare import prepare_splits, train_variant
from autoresearch_bench import build_benchmark, score_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hex_dataset.pt"
OUT_DIR = ROOT / "results"
FIG_DIR = ROOT.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


class PlanarDiskLatentSiteCGNet(nn.Module):
    """General rigid-body Latent-Site Neural CG Potential initialized gently on a 2D disk
    (no hexagon radius or LJ functional form assumed — learns BOTH the 3D site positions
    and the neural radial interaction potential purely from COM + quaternion trajectories!).
    """

    def __init__(self, n_sites: int = 7, hidden: int = 48, init_radius: float = 0.85, r_cut_site: float = 1.45, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # Initialize 1 center + (n_sites - 1) sites on a perturbed ring in xy-plane (NOT exact hexagon!)
        pts = [torch.zeros(3)]
        for k in range(n_sites - 1):
            ang = 2.0 * math.pi * k / (n_sites - 1) + 0.12 * torch.randn(1, generator=gen).item()
            rad = init_radius * (1.0 + 0.12 * torch.randn(1, generator=gen).item())
            z = 0.05 * torch.randn(1, generator=gen).item()
            pts.append(torch.tensor([rad * math.cos(ang), rad * math.sin(ang), z]))
        self.sites = nn.Parameter(torch.stack(pts, dim=0))
        self.r_cut_site = r_cut_site
        self.r_on_site = 0.90

        # Smooth neural radial potential phi(d) = f_cut(d) * (softplus(MLP) ** 2)
        self.register_buffer("rbf_c", torch.linspace(0.65, r_cut_site, 8))
        self.rad_net = nn.Sequential(
            nn.Linear(11, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.rad_net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def forward(self, dx: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        s = self.sites - self.sites.mean(dim=0, keepdim=True)
        R1 = quat_to_rotmat(q1)
        R2 = quat_to_rotmat(q2)
        p1 = torch.einsum("bij,kj->bki", R1, s)
        p2 = torch.einsum("bij,kj->bki", R2, s) + dx[:, None, :]
        diff = p2[:, None, :, :] - p1[:, :, None, :]
        d = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6)

        env = smooth_cutoff(d, self.r_on_site, self.r_cut_site)
        inv_d = 1.0 / d.clamp(min=0.4)
        rbf = torch.exp(-((d.unsqueeze(-1) - self.rbf_c) / 0.14) ** 2)
        feats = torch.cat([d.unsqueeze(-1), inv_d.unsqueeze(-1), (inv_d ** 2).unsqueeze(-1), rbf], dim=-1)
        raw = self.rad_net(feats).squeeze(-1)
        v_pair = env * (F.softplus(raw) ** 2)
        return v_pair.sum(dim=(-1, -2))


def main() -> None:
    torch.set_num_threads(6)
    t_all = time.time()
    d = torch.load(DATA_PATH, weights_only=True)
    splits = prepare_splits(d)
    bench = build_benchmark(d, n_windows=10, n_steps=200)
    exact_floor = score_model(exact_hex_potential, bench)["D"]
    print(f"Autoresearch v2 ready | Exact LJ floor D = {exact_floor:.5f} | Trivial V=0 D = 1.0000")

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
            f"D = {sc['D']:.5f} (dx={sc['dx_mae']:.5f}, dv={sc['dv_mae']:.4f}, L={sc['L_mae']:.4f}, deg={sc['deg_mae']:.3f}°)"
        )
        history.append({"iter": it, "short": short, "desc": desc, "kept": kept, "exact_floor": exact_floor, **sc})

    # Iter 0: Paper Baseline
    print("\n[Iter 0] Paper Baseline (raw 11D MLP, uniform sampling)...")
    m0, _ = train_variant("baseline", splits, epochs=350, batch_size=512, seed=42)
    record(0, "Paper Baseline", "Raw 11D MLP, final-state MSE", m0)

    # Iter 1: SO(3) + C2 Cutoff
    print("\n[Iter 1] SO(3) + C2 Cutoff (ImprovedCGNet)...")
    m1, _ = train_variant("improved", splits, epochs=500, batch_size=512, seed=42)
    record(1, "SO(3)+Cutoff", "16D SO(3) MLP + C2 cutoff + softplus", m1)

    # Iter 2: D6 + Twist + RBF ResNet + Impulse Sampling
    print("\n[Iter 2] ResD6 + Twist + RBF + Impulse-Weighted Sampling...")
    m2 = train_ode_candidate(
        ResD6TwistCGNet(hidden=96), splits, imp_weights=imp_weights,
        epochs=500, lr=3e-3, batch_size=512, horizons=(8, 14, 20), use_huber=True, seed=102, tag="Iter2-ResD6",
    )
    record(2, "ResD6+Twist+RBF", "D6 + twist + RBF ResNet + impulse sampling", m2)

    # Iter 3: Unconstrained 3D Random LatentSite (ablation -> expected discard!)
    print("\n[Iter 3] Random 3D LatentSite (K=7, unconstrained 3D init)...")
    m3 = train_ode_candidate(
        LatentSiteCGNet(n_sites=7, hidden=48, seed=103), splits, imp_weights=imp_weights,
        epochs=400, lr=8e-3, batch_size=512, horizons=(6, 12, 18), use_huber=False, seed=103, tag="Iter3-Rand3D",
    )
    record(3, "3D Rand Sites", "K=7 random 3D init (local min)", m3)

    # Iter 4: Planar-Disk Initialized LatentSite (K=7, learns 3D sites + radial NN from trajectories!)
    print("\n[Iter 4] Planar-Disk LatentSiteCGNet (K=7 learnable 3D sites + radial NN)...")
    m4 = train_ode_candidate(
        PlanarDiskLatentSiteCGNet(n_sites=7, hidden=48, init_radius=0.85, seed=104),
        splits, imp_weights=imp_weights,
        epochs=600, lr=8e-3, batch_size=512, horizons=(6, 12, 20), use_huber=False, seed=104, tag="Iter4-Disk7",
    )
    record(4, "LatentSite (K=7)", "K=7 learnable sites + radial NN", m4)

    # Iter 5: Longer-horizon fine-tune of K=7 LatentSite (T = 15 -> 25 -> 35)
    print("\n[Iter 5] Long-horizon fine-tuning of K=7 LatentSite...")
    m5 = PlanarDiskLatentSiteCGNet(n_sites=7, hidden=48, init_radius=0.85, seed=104)
    m5.load_state_dict(m4.state_dict())
    m5 = train_ode_candidate(
        m5, splits, imp_weights=imp_weights,
        epochs=500, lr=2e-3, batch_size=512, horizons=(15, 25, 35), use_huber=False, seed=105, tag="Iter5-LongH",
    )
    record(5, "K=7 Long-Horizon", "Fine-tuned on T=15..35 rollouts", m5)

    # Iter 6: Precision low-LR polish (batch=768, lr=4e-4)
    print("\n[Iter 6] Precision low-LR polish...")
    m6 = PlanarDiskLatentSiteCGNet(n_sites=7, hidden=48, init_radius=0.85, seed=104)
    m6.load_state_dict(m5.state_dict())
    m6 = train_ode_candidate(
        m6, splits, imp_weights=imp_weights,
        epochs=400, lr=4e-4, batch_size=768, horizons=(20, 30, 40), use_huber=False, seed=106, tag="Iter6-Polish",
    )
    record(6, "Precision Polish", "Low-LR long-horizon polish", m6)

    if isinstance(best_model, PlanarDiskLatentSiteCGNet):
        s = (best_model.sites - best_model.sites.mean(dim=0, keepdim=True)).detach()
        radii = torch.linalg.norm(s, dim=-1).sort().values.tolist()
        print(f"\nDiscovered 3D site radii from COM (ground truth = [0, 1, 1, 1, 1, 1, 1]): {[round(r, 3) for r in radii]}")

    plot_autoresearch_progress(history, best_model, bench, FIG_DIR / "autoresearch_progress.png")
    with open(OUT_DIR / "autoresearch_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nAutoresearch v2 complete in {time.time() - t_all:.1f}s! Best D = {best_D:.5f} (exact floor = {exact_floor:.5f})")


if __name__ == "__main__":
    main()
