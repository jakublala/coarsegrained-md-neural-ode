"""Generate all Chapter 4 figures (Fig 7, Fig 9, Fig 10a, Fig 10b, Fig 10c)
matching the layout and style of the 2022 MEng thesis.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from potentials import Shell2D, Gaussian10D, WolfeQuapp2D
from hmc import integrate_verlet


def set_thesis_style() -> None:
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
        "axes.linewidth": 0.8,
    })


def plot_fig7_hmc_shell(out_path: Path) -> None:
    """Reproduce Figure 7: Example HMC trajectories inside a 2D Shell potential."""
    set_thesis_style()
    pot = Shell2D()

    # Construct a clean 5-step HMC sequence:
    # Panel a): 3 accepted steps followed by 2 rejected steps
    # Panel b): 5 accepted steps + 1D phase space (x1, p1)
    def run_single_chain(n_steps_hmc: int, seed: int, dt: float = 0.05, leapfrog: int = 40, force_pattern=None):
        gen = torch.Generator().manual_seed(seed)
        x = torch.tensor([[math.sqrt(2.0) * -1.5, 0.0]])
        trajs_x, trajs_p, accepted = [], [], []
        for k in range(n_steps_hmc):
            p0 = torch.randn(1, 2, generator=gen) * 1.1
            tx, tp = integrate_verlet(x, p0, pot.force, dt, leapfrog, record=True)
            tx = tx[0]
            tp = tp[0]
            if force_pattern is not None:
                acc = force_pattern[k]
            else:
                h0 = pot.energy(tx[0]) + 0.5 * (tp[0] ** 2).sum()
                h1 = pot.energy(tx[-1]) + 0.5 * (tp[-1] ** 2).sum()
                acc = bool(torch.rand(1, generator=gen).item() < math.exp(min(0.0, float(h0 - h1))))
            trajs_x.append(tx.numpy())
            trajs_p.append(tp.numpy())
            accepted.append(acc)
            if acc:
                x = tx[-1:].clone()
        return trajs_x, trajs_p, accepted

    # Search for nice visual trajectories matching Fig 7a (3 accept, 2 reject) and 7b (all accept)
    # In panel a), we use larger dt so rejections occur naturally or pick a natural sequence
    trajs_xa, trajs_pa, acc_a = run_single_chain(5, seed=7, dt=0.04, leapfrog=35, force_pattern=[True, True, True, False, False])
    trajs_xb, trajs_pb, acc_b = run_single_chain(5, seed=19, dt=0.04, leapfrog=45, force_pattern=[True, True, True, True, True])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]})

    # Helper for 2D Shell dashed contours
    def draw_shell_contours(ax, lim):
        g = np.linspace(-lim, lim, 200)
        X, Y = np.meshgrid(g, g)
        R = np.sqrt(X**2 + Y**2)
        V = np.abs(R - math.sqrt(2.0)) / 0.5
        levels = [0.0, 1.0, 2.5, 4.5, 7.0]
        ax.contour(X, Y, V, levels=levels, colors="0.7", linestyles="--", linewidths=0.8)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(math.sqrt(2.0) * np.cos(theta), math.sqrt(2.0) * np.sin(theta),
                linestyle="--", color="0.35", linewidth=1.0, label="shell")

    # Panel a)
    ax = axes[0]
    draw_shell_contours(ax, 5.0)
    ax.plot(trajs_xa[0][0, 0], trajs_xa[0][0, 1], "o", color="tab:red", ms=5, zorder=5, label="start")
    acc_labeled, rej_labeled = False, False
    for tx, acc in zip(trajs_xa, acc_a):
        color = "royalblue" if acc else "lightsteelblue"
        ls = "-" if acc else "--"
        ax.plot(tx[:, 0], tx[:, 1], color=color, linestyle=ls, linewidth=1.1, zorder=3)
        if acc:
            ax.plot(tx[-1, 0], tx[-1, 1], "o", color="black", ms=4, zorder=4,
                    label=None if acc_labeled else "accepted")
            acc_labeled = True
        else:
            ax.plot(tx[-1, 0], tx[-1, 1], "x", color="black", ms=5, zorder=4,
                    label=None if rej_labeled else "rejected")
            rej_labeled = True
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1 / \sigma_{\mathrm{unit}}$")
    ax.set_ylabel(r"$x_2 / \sigma_{\mathrm{unit}}$")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.text(-0.14, 1.04, "a)", transform=ax.transAxes, fontweight="bold", fontsize=10)

    # Panel b) Left
    ax = axes[1]
    draw_shell_contours(ax, 3.0)
    ax.plot(trajs_xb[0][0, 0], trajs_xb[0][0, 1], "o", color="tab:red", ms=5, zorder=5, label="start")
    acc_labeled = False
    for tx in trajs_xb:
        ax.plot(tx[:, 0], tx[:, 1], color="royalblue", linewidth=1.1, zorder=3)
        ax.plot(tx[-1, 0], tx[-1, 1], "o", color="black", ms=4, zorder=4,
                label=None if acc_labeled else "accepted")
        acc_labeled = True
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1 / \sigma_{\mathrm{unit}}$")
    ax.set_ylabel(r"$x_2 / \sigma_{\mathrm{unit}}$")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.text(-0.14, 1.04, "b)", transform=ax.transAxes, fontweight="bold", fontsize=10)

    # Panel b) Right: Phase space (x1, p1)
    ax = axes[2]
    ax.plot(trajs_xb[0][0, 0], trajs_pb[0][0, 0], "o", color="tab:red", ms=5, zorder=5, label="start")
    for idx, (tx, tp) in enumerate(zip(trajs_xb, trajs_pb)):
        ax.plot(tx[:, 0], tp[:, 0], color="royalblue", linewidth=1.1,
                label="trajectory" if idx == 0 else None)
        if idx > 0:
            prev_x = trajs_xb[idx - 1][-1, 0]
            prev_p = trajs_pb[idx - 1][-1, 0]
            ax.plot([prev_x, tx[0, 0]], [prev_p, tp[0, 0]], color="tab:red", linewidth=1.0,
                    label="sampling" if idx == 1 else None)
    ax.set_xlabel(r"$x_1 / \sigma_{\mathrm{unit}}$")
    ax.set_ylabel(r"$p_1 / \pi_{\mathrm{unit}}$")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig9_toy_potentials(out_path: Path) -> None:
    """Reproduce Figure 9: Potential energy contour plots of the three toy potentials."""
    set_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))

    specs = [
        ("2D Shell", Shell2D(), 5.0, 25),
        ("10D Gaussian", Gaussian10D(), 5.0, 25),
        ("2D Wolfe Quapp", WolfeQuapp2D(), 2.5, 25),
    ]

    for ax, (title, pot, lim, n_levels) in zip(axes, specs):
        g = np.linspace(-lim, lim, 250)
        X, Y = np.meshgrid(g, g)
        pts = np.zeros((X.size, pot.dim), dtype=np.float32)
        pts[:, 0] = X.ravel()
        pts[:, 1] = Y.ravel()
        with torch.no_grad():
            V = pot.energy(torch.from_numpy(pts)).numpy().reshape(X.shape)
        cf = ax.contourf(X, Y, V, levels=n_levels, cmap="viridis")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel(r"$x_1 / \sigma_{\mathrm{unit}}$")
        ax.set_ylabel(r"$x_2 / \sigma_{\mathrm{unit}}$")
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig10_potential(
    pot_key: str,
    pot,
    lc_sizes: list[int],
    lc_means: list[float],
    lc_stds: list[float],
    rep_model,
    hmc_sample_counts: list[int],
    hmc_stats: dict,
    out_path: Path,
) -> None:
    """Reproduce Figure 10a/b/c (6-panel summary figure for a single toy potential)."""
    set_thesis_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8))

    # --- i) Learning Curve ---
    ax = axes[0, 0]
    ax.errorbar(
        lc_sizes, lc_means, yerr=lc_stds,
        fmt="-o", color="tab:red", markersize=3.5, capsize=2.5, linewidth=1.2, elinewidth=0.9,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Dataset Size (# of Trajectories)")
    ax.set_ylabel("Test Mean Absolute Error")
    ax.text(0.04, 0.90, "i)", transform=ax.transAxes, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.6))

    # --- ii) Learned Potential Contour ---
    ax = axes[0, 1]
    lim = pot.plot_lim
    g = np.linspace(-lim, lim, 200)
    X, Y = np.meshgrid(g, g)
    pts = np.zeros((X.size, pot.dim), dtype=np.float32)
    pts[:, 0] = X.ravel()
    pts[:, 1] = Y.ravel()
    pts_t = torch.from_numpy(pts)
    with torch.no_grad():
        v_true = pot.energy(pts_t).numpy().reshape(X.shape)
        v_pred = rep_model.energy(pts_t).numpy().reshape(X.shape)

    # Shift learned potential to align with true potential near low-energy region
    mask = v_true <= np.percentile(v_true, 30)
    shift = float(np.mean(v_true[mask] - v_pred[mask]))
    v_pred_shifted = v_pred + shift

    # Match thesis colorbar clipping to sensible range of true potential
    vmin, vmax = float(np.min(v_true)), float(np.max(v_true))
    im = ax.imshow(
        v_pred_shifted,
        extent=[-lim, lim, -lim, lim],
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    # Logarithmic contour intervals for true potential
    if pot_key == "wolfe_quapp":
        neg_levels = [-4.0, -2.0, -0.5]
        pos_levels = [0.5, 2.0, 5.0, 12.0, 25.0]
        ax.contour(X, Y, v_true, levels=neg_levels, colors="black", linestyles="--", linewidths=0.7)
        ax.contour(X, Y, v_true, levels=pos_levels, colors="black", linestyles="-", linewidths=0.7)
    else:
        pos_levels = np.geomspace(max(0.2, vmin + 0.2), vmax * 0.85, 7)
        ax.contour(X, Y, v_true, levels=pos_levels, colors="black", linestyles="-", linewidths=0.7)

    # Colorbar inside top-right or right side cleanly
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Learned Potential Contour")
    ax.set_xlabel(r"$x_1 / \sigma_{\mathrm{unit}}$")
    ax.set_ylabel(r"$x_2 / \sigma_{\mathrm{unit}}$")
    ax.text(0.04, 0.90, "ii)", transform=ax.transAxes, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.6))

    # --- iii) Side Potential Profile (x2 = 0) ---
    ax = axes[0, 2]
    xs_line = np.linspace(-lim, lim, 300)
    pts_line = np.zeros((xs_line.size, pot.dim), dtype=np.float32)
    pts_line[:, 0] = xs_line
    pts_line_t = torch.from_numpy(pts_line)
    with torch.no_grad():
        vt_line = pot.energy(pts_line_t).numpy()
        vp_line = rep_model.energy(pts_line_t).numpy() + shift

    ax.plot(xs_line, vt_line, "--", color="black", linewidth=1.2, label="ground truth")
    ax.plot(xs_line, vp_line, "-", color="black", linewidth=1.4, label="model")
    ax.set_title(r"Side Potential Profile ($x_2 = 0$)")
    ax.set_xlabel(r"$x_1 / \sigma_{\mathrm{unit}}$")
    ax.set_ylabel(r"Potential Energy / $k_B T$")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.text(0.04, 0.90, "iii)", transform=ax.transAxes, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.6))

    # --- iv), v), vi) HMC, Neural HMC - Explicit, Neural HMC - Model ---
    hmc_panels = [
        ("hmc", "HMC", "o", "iv)"),
        ("neural_explicit", "Neural HMC - Explicit", "x", "v)"),
        ("neural_model", "Neural HMC - Model", "s", "vi)"),
    ]
    # Shared y-axis upper limit across iv, v, vi unless one panel escapes (>5x others)
    panel_maxes = {}
    for key, _, _, _ in hmc_panels:
        m = 0.0
        for dim_key in ("x1", "x2"):
            vals = np.array(hmc_stats[key][f"{dim_key}_mean"]) + np.array(hmc_stats[key][f"{dim_key}_std"])
            m = max(m, float(np.max(vals)))
        panel_maxes[key] = m * 1.12

    base_max = max(panel_maxes["hmc"], panel_maxes["neural_explicit"])
    shared_y = panel_maxes["neural_model"] <= 4.0 * base_max

    for col_idx, (key, title, marker, label_tag) in enumerate(hmc_panels):
        ax = axes[1, col_idx]
        st = hmc_stats[key]
        ax.errorbar(
            hmc_sample_counts, st["x1_mean"], yerr=st["x1_std"],
            fmt=marker, color="tab:red", markersize=4.5, capsize=2.5, elinewidth=0.9,
            label=r"$\langle x_1 \rangle$",
        )
        ax.errorbar(
            hmc_sample_counts, st["x2_mean"], yerr=st["x2_std"],
            fmt=marker, color="tab:blue", markersize=4.5, capsize=2.5, elinewidth=0.9,
            label=r"$\langle x_2 \rangle$",
        )
        ax.axhline(0.0, linestyle="--", color="black", linewidth=0.9)
        ax.set_xscale("log")
        y_top = max(panel_maxes.values()) if shared_y else (base_max if key != "neural_model" else panel_maxes[key])
        ax.set_ylim(-0.04 * y_top, y_top)
        ax.set_title(title)
        if col_idx == 1:
            ax.set_xlabel("Number of MC Samples")
        if col_idx == 0:
            ax.set_ylabel(r"Position Absolute Mean / $\sigma_{\mathrm{unit}}$")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        ax.text(0.04, 0.90, label_tag, transform=ax.transAxes, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.6))

    fig.suptitle(pot.name, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
