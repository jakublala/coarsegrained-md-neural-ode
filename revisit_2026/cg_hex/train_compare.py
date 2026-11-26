"""Train and compare three CG hexagon Neural ODE models on LAMMPS trajectories:
  1. baseline:    EUROCON 2023 setup (68-48-32 tanh MLP, raw 11D, final-state MSE, uniform sampling, no reg)
  2. regularized: Same raw 11D MLP + L2 weight decay + temporal regularization (STEER)
  3. improved:    SO(3) + q==-q + 1<->2 symmetric MLP + C^2 smooth cutoff + softplus,
                  collision-weighted stratified batching, normalized multi-step + chordal quat loss.

Generates:
  - figures/cg_fig4_test_trajectory.png  (Held-out TEST trajectory comparison across a real collision)
  - figures/cg_fig5_energy_landscape.png (Exact vs Baseline vs Regularized vs Improved 3D landscapes)
  - results/cg_comparison.json           (Quantitative test metrics & energy accuracy)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rigid_ode import (
    RawMLPNet,
    ImprovedCGNet,
    exact_hex_potential,
    rollout_rigid_body,
    quat_to_rotmat,
    R_MAX,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hex_dataset.pt"
OUT_DIR = ROOT / "results"
FIG_DIR = ROOT.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })


def prepare_splits(d: dict) -> dict:
    """Split 10 LAMMPS trajectories into 8 train / 1 val / 1 test (exact paper 80-10-10 split)."""
    dx = d["x2"] - d["x1"]
    dv = d["v2"] - d["v1"]
    # Subsample or keep full 100k frames per trajectory
    splits = {}
    for name, sl in [("train", slice(0, 8)), ("val", slice(8, 9)), ("test", slice(9, 10))]:
        sub_dx = dx[sl]
        sub_L1 = d["L1"][sl]
        sub_L2 = d["L2"][sl]
        max_t = sub_dx.shape[1] - 65
        # Trajectory-only collision detector: angular momentum is strictly conserved
        # during free flight, so |dL1| + |dL2| > 1e-4 pinpoints actual bead-bead collisions!
        dL = (sub_L1[:, 1 : max_t + 1] - sub_L1[:, :max_t]).norm(dim=-1) + (
            sub_L2[:, 1 : max_t + 1] - sub_L2[:, :max_t]
        ).norm(dim=-1)
        active_mask = dL > 1e-4
        active_idx = torch.nonzero(active_mask, as_tuple=False)  # (K, 2) -> (traj_id, t0)
        # Also index near-contact non-colliding frames (r < R_MAX and dL <= 1e-4)
        sub_r = torch.linalg.norm(sub_dx[:, :max_t], dim=-1)
        near_mask = (sub_r < R_MAX) & (~active_mask)
        near_idx = torch.nonzero(near_mask, as_tuple=False)
        splits[name] = {
            "dx": sub_dx,
            "dv": dv[sl],
            "q1": d["q1"][sl],
            "q2": d["q2"][sl],
            "L1": sub_L1,
            "L2": sub_L2,
            "active_idx": active_idx,
            "near_idx": near_idx,
            "max_t": max_t,
        }

    # Compute change-scales over 20 steps on training active collision frames
    tr = splits["train"]
    idx = tr["active_idx"][:: max(1, len(tr["active_idx"]) // 4000)]
    ti, t0 = idx[:, 0], idx[:, 1]
    t1 = t0 + 20
    q1_dot = (tr["q1"][ti, t1] * tr["q1"][ti, t0]).sum(dim=-1).clamp(-1.0, 1.0)
    scales = {
        "dx": float((tr["dx"][ti, t1] - tr["dx"][ti, t0]).std().clamp(min=1e-4)),
        "dv": float((tr["dv"][ti, t1] - tr["dv"][ti, t0]).std().clamp(min=1e-4)),
        "L": float((tr["L1"][ti, t1] - tr["L1"][ti, t0]).std().clamp(min=1e-4)),
        "q_chord": float((1.0 - q1_dot.pow(2)).mean().clamp(min=1e-6)),
    }
    splits["scales"] = scales
    splits["meta"] = {
        "dt": d["dt"],
        "K_ext": d["K_ext"],
        "r0_ext": d["r0_ext"],
        "mass": d["mass"],
    }
    return splits


def sample_batch(split: dict, batch_size: int, n_steps: int, stratified: bool, gen: torch.Generator) -> dict:
    n_traj = split["dx"].shape[0]
    max_t = split["dx"].shape[1] - n_steps - 1

    if stratified and len(split["active_idx"]) > 0:
        n_act = int(0.75 * batch_size)
        n_near = batch_size - n_act
        sel = torch.randint(0, len(split["active_idx"]), (n_act,), generator=gen)
        # Shift start slightly backward (0..5 steps) so snippets enter into collisions
        jitter = torch.randint(0, 6, (n_act,), generator=gen)
        ti_act = split["active_idx"][sel, 0]
        t0_act = (split["active_idx"][sel, 1] - jitter).clamp(min=0, max=max_t)
        sel_n = torch.randint(0, len(split["near_idx"]), (n_near,), generator=gen)
        ti_near = split["near_idx"][sel_n, 0]
        t0_near = split["near_idx"][sel_n, 1].clamp(max=max_t)
        ti = torch.cat([ti_act, ti_near], dim=0)
        t0 = torch.cat([t0_act, t0_near], dim=0)
    else:
        ti = torch.randint(0, n_traj, (batch_size,), generator=gen)
        t0 = torch.randint(0, max_t + 1, (batch_size,), generator=gen)

    offsets = torch.arange(n_steps + 1).unsqueeze(0)
    t_mat = t0.unsqueeze(1) + offsets  # (B, T+1)
    ti_mat = ti.unsqueeze(1)
    return {
        "dx": split["dx"][ti_mat, t_mat],
        "dv": split["dv"][ti_mat, t_mat],
        "q1": split["q1"][ti_mat, t_mat],
        "q2": split["q2"][ti_mat, t_mat],
        "L1": split["L1"][ti_mat, t_mat],
        "L2": split["L2"][ti_mat, t_mat],
    }


def train_variant(variant: str, splits: dict, epochs: int = 450, batch_size: int = 512, seed: int = 42) -> tuple[nn.Module, list[float]]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    meta = splits["meta"]
    scales = splits["scales"]

    if variant == "baseline":
        model = RawMLPNet()
        opt = torch.optim.Adamax(model.parameters(), lr=0.02)
        stratified = False
    elif variant == "regularized":
        model = RawMLPNet()
        opt = torch.optim.Adamax(model.parameters(), lr=0.02, weight_decay=1e-3)
        stratified = False
    elif variant == "improved":
        model = ImprovedCGNet(hidden=64)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
        stratified = True
    else:
        raise ValueError(variant)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    losses = []
    t_start = time.time()

    for ep in range(epochs):
        # Curriculum horizon: 10 -> 20 -> 30 steps
        if variant == "regularized":
            # Temporal regularization (STEER): randomize integration horizon
            base_h = 10 + 10 * min(2, ep // (epochs // 3))
            n_steps = int(torch.randint(max(5, base_h - 5), base_h + 6, (1,), generator=gen).item())
        else:
            n_steps = 10 + 10 * min(2, ep // (epochs // 3))

        batch = sample_batch(splits["train"], batch_size, n_steps, stratified=stratified, gen=gen)
        opt.zero_grad(set_to_none=True)

        pred = rollout_rigid_body(
            model,
            batch["dx"][:, 0], batch["dv"][:, 0],
            batch["q1"][:, 0], batch["q2"][:, 0],
            batch["L1"][:, 0], batch["L2"][:, 0],
            dt=meta["dt"], n_steps=n_steps,
            K_ext=meta["K_ext"], r0_ext=meta["r0_ext"], mass=meta["mass"],
            create_graph=True, record_all=(variant == "improved"),
        )

        if variant in ("baseline", "regularized"):
            # Paper Eqs. 22-23: unnormalized final-state MSE
            loss = (
                (pred["dx"] - batch["dx"][:, -1]).pow(2).sum(dim=-1)
                + (pred["dv"] - batch["dv"][:, -1]).pow(2).sum(dim=-1)
                + (pred["q1"] - batch["q1"][:, -1]).pow(2).sum(dim=-1)
                + (pred["q2"] - batch["q2"][:, -1]).pow(2).sum(dim=-1)
                + (pred["L1"] - batch["L1"][:, -1]).pow(2).sum(dim=-1)
                + (pred["L2"] - batch["L2"][:, -1]).pow(2).sum(dim=-1)
            ).mean() / 20.0
        else:
            # Improved: normalized multi-step trajectory MSE + sign-invariant chordal quaternion loss
            l_dx = ((pred["dx"] - batch["dx"]) / scales["dx"]).pow(2).mean()
            l_dv = ((pred["dv"] - batch["dv"]) / scales["dv"]).pow(2).mean()
            l_L = 0.5 * (
                ((pred["L1"] - batch["L1"]) / scales["L"]).pow(2).mean()
                + ((pred["L2"] - batch["L2"]) / scales["L"]).pow(2).mean()
            )
            q1_dot = (pred["q1"] * batch["q1"]).sum(dim=-1).clamp(-1.0, 1.0)
            q2_dot = (pred["q2"] * batch["q2"]).sum(dim=-1).clamp(-1.0, 1.0)
            l_q = 0.5 * ((1.0 - q1_dot.pow(2)).mean() + (1.0 - q2_dot.pow(2)).mean()) / scales["q_chord"]
            loss = l_dx + l_dv + l_L + 0.5 * l_q

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        if variant == "improved":
            scheduler.step()

        losses.append(float(loss.item()))
        if (ep + 1) % 150 == 0 or ep == 0:
            print(f"  [{variant:11s}] ep {ep+1:3d}/{epochs} | n_steps={n_steps:2d} | loss={loss.item():.5f} ({time.time()-t_start:.1f}s)")

    return model, losses


@torch.no_grad()
def evaluate_model(model: nn.Module, splits: dict, label: str) -> dict:
    """Evaluate on held-out TEST set:
    1. 50-step collision rollout accuracy (200 random test collision snippets)
    2. Direct potential energy accuracy vs exact 7-bead LJ on 2,000 test collision frames
       (after optimal constant shift, since potentials are defined up to an additive constant).
    """
    model.eval()
    meta = splits["meta"]
    te = splits["test"]
    gen = torch.Generator().manual_seed(999)

    # 1. Rollout metrics on 200 test collision windows (50 steps each)
    batch = sample_batch(te, 200, 50, stratified=True, gen=gen)
    # Note: compute_forces_and_torques needs autograd even in eval, so enable_grad locally
    with torch.enable_grad():
        pred = rollout_rigid_body(
            model,
            batch["dx"][:, 0], batch["dv"][:, 0],
            batch["q1"][:, 0], batch["q2"][:, 0],
            batch["L1"][:, 0], batch["L2"][:, 0],
            dt=meta["dt"], n_steps=50,
            K_ext=meta["K_ext"], r0_ext=meta["r0_ext"], mass=meta["mass"],
            create_graph=False, record_all=True,
        )
    dx_mae = float((pred["dx"] - batch["dx"]).abs().mean().item())
    dv_mae = float((pred["dv"] - batch["dv"]).abs().mean().item())
    L_mae = float(0.5 * ((pred["L1"] - batch["L1"]).abs().mean() + (pred["L2"] - batch["L2"]).abs().mean()).item())
    q_err = float(0.5 * (
        (1.0 - (pred["q1"] * batch["q1"]).sum(dim=-1).abs().clamp(max=1.0)).mean()
        + (1.0 - (pred["q2"] * batch["q2"]).sum(dim=-1).abs().clamp(max=1.0)).mean()
    ).item())

    # 2. Direct energy correlation & MAE on 2,000 test configurations (active + outer)
    idx_e = torch.randint(0, te["dx"].shape[1], (2000,), generator=gen)
    dx_e = te["dx"][0, idx_e]
    q1_e = te["q1"][0, idx_e]
    q2_e = te["q2"][0, idx_e]
    v_true = exact_hex_potential(dx_e, q1_e, q2_e)
    v_pred = model(dx_e, q1_e, q2_e)
    # Align offset using non-interacting frames (r > R_MAX + 0.2)
    outer = torch.linalg.norm(dx_e, dim=-1) > (R_MAX + 0.1)
    shift = float((v_true[outer] - v_pred[outer]).mean().item()) if outer.any() else float((v_true - v_pred).mean().item())
    v_pred_s = v_pred + shift
    v_mae = float((v_true - v_pred_s).abs().mean().item())
    v_outer_std = float(v_pred_s[outer].abs().mean().item()) if outer.any() else 0.0
    corr = float(np.corrcoef(v_true.numpy(), v_pred_s.numpy())[0, 1]) if v_pred_s.std() > 1e-6 else 0.0

    res = {
        "dx_mae_50": dx_mae,
        "dv_mae_50": dv_mae,
        "L_mae_50": L_mae,
        "q_err_50": q_err,
        "energy_mae": v_mae,
        "outer_spurious_mae": v_outer_std,
        "energy_corr": corr,
        "shift": shift,
    }
    print(f"  Eval [{label:11s}] TEST 50-step: dx={dx_mae:.5f}, dv={dv_mae:.5f}, L={L_mae:.5f}, q={q_err:.2e} | Energy MAE={v_mae:.4f}, OuterSpurious={v_outer_std:.4f}, Corr={corr:.4f}")
    return res


def plot_test_trajectory_comparison(models: dict[str, nn.Module], splits: dict, out_path: Path) -> None:
    """Reproduce Fig. 4 style plot — but on a HELD-OUT TEST collision trajectory (1,500 steps = 1.5 tau)!"""
    set_style()
    meta = splits["meta"]
    te = splits["test"]
    dx_all = te["dx"][0]
    r_all = torch.linalg.norm(dx_all, dim=-1)

    # Find a dramatic collision in the test trajectory where r dips below 2.1
    cand = torch.nonzero(r_all[1000:50000] < 2.1, as_tuple=False)
    t_peak = int(cand[0].item()) + 1000 if len(cand) > 0 else int(torch.argmin(r_all[1000:50000]).item()) + 1000
    t0 = max(0, t_peak - 500)
    n_steps = 1500
    t_axis = np.arange(n_steps + 1) * meta["dt"]

    true_dx = te["dx"][0, t0 : t0 + n_steps + 1]
    true_dv = te["dv"][0, t0 : t0 + n_steps + 1]
    true_L1 = te["L1"][0, t0 : t0 + n_steps + 1]
    true_q1 = te["q1"][0, t0 : t0 + n_steps + 1]

    preds = {}
    for name, model in models.items():
        model.eval()
        with torch.enable_grad():
            preds[name] = rollout_rigid_body(
                model,
                te["dx"][0:1, t0], te["dv"][0:1, t0],
                te["q1"][0:1, t0], te["q2"][0:1, t0],
                te["L1"][0:1, t0], te["L2"][0:1, t0],
                dt=meta["dt"], n_steps=n_steps,
                K_ext=meta["K_ext"], r0_ext=meta["r0_ext"], mass=meta["mass"],
                create_graph=False, record_all=True,
            )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), sharex=True)
    colors = {"baseline": "tab:red", "regularized": "tab:orange", "improved": "tab:blue"}
    labels = {"baseline": "Paper Baseline (11D raw)", "regularized": "Paper + L2/STEER", "improved": "Improved (SO(3)+Cutoff)"}

    # Panel a): Relative separation ||x2 - x1||
    ax = axes[0, 0]
    ax.plot(t_axis, torch.linalg.norm(true_dx, dim=-1).numpy(), "--", color="black", lw=1.6, label="Ground truth (LAMMPS)")
    for name in ("baseline", "regularized", "improved"):
        r_p = torch.linalg.norm(preds[name]["dx"][0].detach(), dim=-1).numpy()
        ax.plot(t_axis, r_p, color=colors[name], lw=1.3, label=labels[name], alpha=0.9)
    ax.axhline(R_MAX, linestyle=":", color="0.5", lw=0.9, label=r"$r_{\max} = 2l + r_c$")
    ax.set_ylabel(r"$\|\mathbf{x}_2 - \mathbf{x}_1\|\ /\ \sigma_{\mathrm{unit}}$")
    ax.set_title("a) Relative Separation (Held-Out Test Collision)")
    ax.legend(loc="best", frameon=True)

    # Panel b): Relative speed ||v2 - v1||
    ax = axes[0, 1]
    ax.plot(t_axis, torch.linalg.norm(true_dv, dim=-1).numpy(), "--", color="black", lw=1.6, label="LAMMPS")
    for name in ("baseline", "regularized", "improved"):
        v_p = torch.linalg.norm(preds[name]["dv"][0].detach(), dim=-1).numpy()
        ax.plot(t_axis, v_p, color=colors[name], lw=1.3, label=labels[name], alpha=0.9)
    ax.set_ylabel(r"$\|\mathbf{v}_2 - \mathbf{v}_1\|\ /\ (\sigma_{\mathrm{unit}}\tau_{\mathrm{unit}}^{-1})$")
    ax.set_title("b) Relative Speed")

    # Panel c): Body 1 angular momentum magnitude ||L1||
    ax = axes[1, 0]
    ax.plot(t_axis, torch.linalg.norm(true_L1, dim=-1).numpy(), "--", color="black", lw=1.6, label="LAMMPS")
    for name in ("baseline", "regularized", "improved"):
        L_p = torch.linalg.norm(preds[name]["L1"][0].detach(), dim=-1).numpy()
        ax.plot(t_axis, L_p, color=colors[name], lw=1.3, label=labels[name], alpha=0.9)
    ax.set_xlabel(r"Time / $\tau_{\mathrm{unit}}$")
    ax.set_ylabel(r"$\|\mathbf{L}_1\|$")
    ax.set_title("c) Body 1 Angular Momentum Magnitude")

    # Panel d): Body 1 orientation error (degrees) vs LAMMPS
    ax = axes[1, 1]
    for name in ("baseline", "regularized", "improved"):
        q_p = preds[name]["q1"][0].detach()
        dot = (q_p * true_q1).sum(dim=-1).abs().clamp(0.0, 1.0)
        deg = (2.0 * torch.acos(dot) * (180.0 / np.pi)).numpy()
        ax.plot(t_axis, deg, color=colors[name], lw=1.3, label=labels[name], alpha=0.9)
    ax.set_xlabel(r"Time / $\tau_{\mathrm{unit}}$")
    ax.set_ylabel("Orientation Error (deg)")
    ax.set_title("d) Body 1 Orientation Error vs. LAMMPS")

    fig.suptitle("Held-Out Test Trajectory Rollout (1,500 Steps Across a Real Collision)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig5_landscapes(models: dict[str, nn.Module], shifts: dict[str, float], out_path: Path) -> None:
    """Reproduce Fig. 5 (in-plane xy potential energy surface) comparing Exact vs all 3 models."""
    set_style()
    g = np.linspace(-4.0, 4.0, 75)
    X, Y = np.meshgrid(g, g)
    N = X.size
    dx = torch.zeros(N, 3)
    dx[:, 0] = torch.from_numpy(X.ravel().astype(np.float32))
    dx[:, 1] = torch.from_numpy(Y.ravel().astype(np.float32))
    q_id = torch.zeros(N, 4)
    q_id[:, 0] = 1.0

    with torch.no_grad():
        v_exact = exact_hex_potential(dx, q_id, q_id).clamp(max=6.0).numpy().reshape(X.shape)
        surfaces = [("Exact 7-Bead LJ (Ground Truth)", v_exact)]
        titles = {
            "baseline": "Paper Baseline (11D Raw, No Reg)",
            "regularized": "Paper + L2 & STEER Regularization",
            "improved": "Improved (SO(3) + C² Cutoff)",
        }
        for key in ("baseline", "regularized", "improved"):
            m = models[key]
            m.eval()
            v_p = (m(dx, q_id, q_id) + shifts[key]).clamp(min=-1.0, max=6.0).numpy().reshape(X.shape)
            surfaces.append((titles[key], v_p))

    fig = plt.figure(figsize=(13.0, 6.2))
    for i, (title, Z) in enumerate(surfaces):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0.1, antialiased=True, vmin=0.0, vmax=6.0)
        ax.set_title(title, fontsize=9.5, pad=6)
        ax.set_xlabel(r"$\Delta x_1\ (\sigma_{\mathrm{unit}})$", labelpad=4)
        ax.set_ylabel(r"$\Delta x_2\ (\sigma_{\mathrm{unit}})$", labelpad=4)
        ax.set_zlabel(r"$V_{CG}\ (\epsilon_{\mathrm{unit}})$", labelpad=2)
        ax.set_zlim(-0.5, 6.2)
        ax.view_init(elev=30, azim=-50)

    fig.suptitle("Figure 5 Comparison: In-Plane Coarse-Grained Potential Energy Landscape", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    t_all = time.time()
    print("Loading LAMMPS dataset...")
    d = torch.load(DATA_PATH, weights_only=True)
    splits = prepare_splits(d)
    print(f"  Train active collision frames: {len(splits['train']['active_idx'])} / {8 * splits['train']['max_t']}")
    print(f"  Change scales (20 steps): {splits['scales']}")

    models = {}
    metrics = {}
    shifts = {}

    for variant in ("baseline", "regularized", "improved"):
        print(f"\n=== Training {variant} ===")
        ep = 750 if variant == "improved" else 450
        model, hist = train_variant(variant, splits, epochs=ep, batch_size=512, seed=42)
        torch.save(model.state_dict(), OUT_DIR / f"cg_{variant}.pt")
        models[variant] = model
        ev = evaluate_model(model, splits, variant)
        metrics[variant] = ev
        shifts[variant] = ev["shift"]

    print("\nGenerating comparison figures...")
    plot_test_trajectory_comparison(models, splits, FIG_DIR / "cg_fig4_test_trajectory.png")
    plot_fig5_landscapes(models, shifts, FIG_DIR / "cg_fig5_energy_landscape.png")

    with open(OUT_DIR / "cg_comparison.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone in {time.time() - t_all:.1f}s!")


if __name__ == "__main__":
    main()
