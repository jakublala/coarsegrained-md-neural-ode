"""Render a synchronized side-by-side MP4 video comparing:
  - Left 3D panel:  Original Fully Atomistic LAMMPS Simulation (14 point particles, 2 hexagons)
  - Right 3D panel: Coarse-Grained Neural ODE Simulation (COM + 3D orientation arrows + discovered latent sites)
  - Bottom panels:  Live synchronized Separation ||dx(t)|| and Angular Momentum ||L1(t)|| traces
                    using REAL held-out test trajectories (traj_10) and our winning autoresearch_best.pt!
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rigid_ode import HEX_LOCAL, quat_to_rotmat, rollout_rigid_body
from run_autoresearch import LatentSiteCGNet

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hex_dataset.pt"
MODEL_PATH = ROOT / "results" / "autoresearch_best.pt"
OUT_MP4 = ROOT.parent / "figures" / "atomistic_vs_cg_long.mp4"


def main() -> None:
    d = torch.load(DATA_PATH, weights_only=True)
    model = LatentSiteCGNet(n_sites=7, hidden=48, seed=103)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # Use held-out test trajectory (traj_10, index 9) across a long multi-collision segment
    dx_all = d["x2"][9] - d["x1"][9]
    dv_all = d["v2"][9] - d["v1"][9]
    q1_all, q2_all = d["q1"][9], d["q2"][9]
    L1_all, L2_all = d["L1"][9], d["L2"][9]

    # Pick a long 15,000-step (15.0 tau) multi-collision window in traj_10
    dL = (L1_all[1:] - L1_all[:-1]).norm(dim=-1)
    peaks = torch.nonzero(dL[5000:40000] > 0.015, as_tuple=False)
    t_peak = int(peaks[0].item()) + 5000 if len(peaks) > 0 else 10000
    t0 = max(0, t_peak - 500)
    n_steps = 15000

    # Roll out the winning Coarse-Grained Neural ODE from the exact same initial state
    with torch.enable_grad():
        pred = rollout_rigid_body(
            model,
            dx_all[t0 : t0 + 1], dv_all[t0 : t0 + 1],
            q1_all[t0 : t0 + 1], q2_all[t0 : t0 + 1],
            L1_all[t0 : t0 + 1], L2_all[t0 : t0 + 1],
            dt=d["dt"], n_steps=n_steps,
            K_ext=d["K_ext"], r0_ext=d["r0_ext"], mass=d["mass"],
            create_graph=False, record_all=True,
        )

    # Extract numpy arrays in center-of-mass frame (x1 = -dx/2, x2 = +dx/2)
    dx_true = dx_all[t0 : t0 + n_steps + 1]
    q1_true = q1_all[t0 : t0 + n_steps + 1]
    q2_true = q2_all[t0 : t0 + n_steps + 1]
    L1_true = L1_all[t0 : t0 + n_steps + 1]

    dx_cg = pred["dx"][0].detach()
    q1_cg = pred["q1"][0].detach()
    q2_cg = pred["q2"][0].detach()
    L1_cg = pred["L1"][0].detach()

    # Precompute 3D atom positions & CG site positions for all frames
    R1_t = quat_to_rotmat(q1_true)
    R2_t = quat_to_rotmat(q2_true)
    R1_c = quat_to_rotmat(q1_cg)
    R2_c = quat_to_rotmat(q2_cg)

    pts_true = HEX_LOCAL  # (7, 3)
    sites_cg = (model.sites - model.sites.mean(dim=0, keepdim=True)).detach()  # (7, 3)

    x1_t = -0.5 * dx_true
    x2_t = +0.5 * dx_true
    x1_c = -0.5 * dx_cg
    x2_c = +0.5 * dx_cg

    atoms1_t = torch.einsum("tij,nj->tni", R1_t, pts_true) + x1_t[:, None, :]
    atoms2_t = torch.einsum("tij,nj->tni", R2_t, pts_true) + x2_t[:, None, :]
    sites1_c = torch.einsum("tij,nj->tni", R1_c, sites_cg) + x1_c[:, None, :]
    sites2_c = torch.einsum("tij,nj->tni", R2_c, sites_cg) + x2_c[:, None, :]

    # Normals (body z-axis) for CG orientation arrows
    n1_c = R1_c[:, :, 2] * 1.15
    n2_c = R2_c[:, :, 2] * 1.15

    t_axis = np.arange(n_steps + 1) * d["dt"]
    r_true_np = dx_true.norm(dim=-1).numpy()
    r_cg_np = dx_cg.norm(dim=-1).numpy()
    L_true_np = L1_true.norm(dim=-1).numpy()
    L_cg_np = L1_cg.norm(dim=-1).numpy()

    # Subsample 300 frames (every 50th step) -> 25 fps = 12.0 seconds covering 15.0 tau
    frame_indices = np.linspace(0, n_steps, 300, dtype=int)

    # Dark sleek scientific styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#161922",
        "axes.edgecolor": "#2e3446",
        "axes.labelcolor": "#d1d5db",
        "xtick.color": "#9ca3af",
        "ytick.color": "#9ca3af",
        "text.color": "#f3f4f6",
    })

    fig = plt.figure(figsize=(11.2, 6.4), dpi=130)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.65, 1.0], hspace=0.28, wspace=0.18, left=0.06, right=0.96, top=0.90, bottom=0.08)
    ax3d_l = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d_r = fig.add_subplot(gs[0, 1], projection="3d")
    ax_sep = fig.add_subplot(gs[1, 0])
    ax_ang = fig.add_subplot(gs[1, 1])

    # Static background traces on bottom plots
    ax_sep.plot(t_axis, r_true_np, "--", color="#e5e7eb", lw=1.6, alpha=0.85, label="Atomistic LAMMPS")
    ax_sep.plot(t_axis, r_cg_np, "-", color="#38bdf8", lw=1.8, alpha=0.9, label="Coarse-Grained Neural ODE")
    ax_sep.set_ylabel(r"Separation $\|\Delta\mathbf{x}\|$ ($\sigma$)")
    ax_sep.set_xlabel(r"Time ($\tau_{\mathrm{unit}}$)")
    ax_sep.set_title("Relative COM Separation (Held-Out Test Collision)", fontsize=9.5, pad=5)
    ax_sep.legend(loc="upper right", frameon=True, facecolor="#1f2430", edgecolor="#374151", fontsize=8)
    ax_sep.grid(color="#2e3446", alpha=0.5)
    cursor_sep = ax_sep.axvline(0.0, color="#22c55e", lw=1.5)
    dot_sep_t, = ax_sep.plot([], [], "o", color="#ffffff", ms=5, zorder=5)
    dot_sep_c, = ax_sep.plot([], [], "o", color="#38bdf8", ms=4.5, zorder=6)

    ax_ang.plot(t_axis, L_true_np, "--", color="#e5e7eb", lw=1.6, alpha=0.85, label="Atomistic LAMMPS")
    ax_ang.plot(t_axis, L_cg_np, "-", color="#fb923c", lw=1.8, alpha=0.9, label="Coarse-Grained Neural ODE")
    ax_ang.set_ylabel(r"Angular Momentum $\|\mathbf{L}_1\|$")
    ax_ang.set_xlabel(r"Time ($\tau_{\mathrm{unit}}$)")
    ax_ang.set_title("Body 1 Angular Momentum Transfer", fontsize=9.5, pad=5)
    ax_ang.legend(loc="upper right", frameon=True, facecolor="#1f2430", edgecolor="#374151", fontsize=8)
    ax_ang.grid(color="#2e3446", alpha=0.5)
    cursor_ang = ax_ang.axvline(0.0, color="#22c55e", lw=1.5)
    dot_ang_t, = ax_ang.plot([], [], "o", color="#ffffff", ms=5, zorder=5)
    dot_ang_c, = ax_ang.plot([], [], "o", color="#fb923c", ms=4.5, zorder=6)

    title_txt = fig.suptitle("", fontsize=11.5, fontweight="bold", y=0.965)

    def setup_3d(ax, title, azim):
        ax.clear()
        ax.set_facecolor("#161922")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_zlim(-2.2, 2.2)
        ax.set_box_aspect([1.0, 1.0, 0.85])
        ax.view_init(elev=22, azim=azim)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#252a38")
        ax.yaxis.pane.set_edgecolor("#252a38")
        ax.zaxis.pane.set_edgecolor("#252a38")
        ax.grid(color="#252a38", linewidth=0.5)

    writer = FFMpegWriter(fps=25, metadata={"title": "Atomistic vs Coarse-Grained Neural ODE"}, bitrate=2600)
    print(f"Rendering {len(frame_indices)} frames to {OUT_MP4}...")
    with writer.saving(fig, str(OUT_MP4), dpi=130):
        for k, f in enumerate(frame_indices):
            azim = -55 + 22 * (k / len(frame_indices))  # gentle cinematic camera orbit
            setup_3d(ax3d_l, "Original Atomistic (LAMMPS · 14 Particles)", azim)
            setup_3d(ax3d_r, "Coarse-Grained Neural ODE (2 COMs + Quaternions)", azim)

            # --- Left: Fully Atomistic (14 beads + hexagonal polygons) ---
            a1 = atoms1_t[f].numpy()
            a2 = atoms2_t[f].numpy()
            for pts, col, ecol in [(a1, "#38bdf8", "#0284c7"), (a2, "#fb923c", "#ea580c")]:
                poly = Poly3DCollection([pts[1:]], alpha=0.22, facecolor=col, edgecolor=col, linewidths=1.2)
                ax3d_l.add_collection3d(poly)
                ax3d_l.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=135, color=col, edgecolors="#0f1117", linewidths=0.8, depthshade=True)
            # Subtle COM spring line
            c1t, c2t = x1_t[f].numpy(), x2_t[f].numpy()
            ax3d_l.plot([c1t[0], c2t[0]], [c1t[1], c2t[1]], [c1t[2], c2t[2]], ":", color="#6b7280", lw=1.0)

            # --- Right: Coarse-Grained (COM spheres + 3D orientation arrows + ghost latent sites) ---
            c1c, c2c = x1_c[f].numpy(), x2_c[f].numpy()
            n1, n2 = n1_c[f].numpy(), n2_c[f].numpy()
            s1, s2 = sites1_c[f].numpy(), sites2_c[f].numpy()
            # Ghost discovered latent sites (translucent, showing the learned effective shape)
            ax3d_r.scatter(s1[:, 0], s1[:, 1], s1[:, 2], s=42, color="#38bdf8", alpha=0.30, depthshade=True)
            ax3d_r.scatter(s2[:, 0], s2[:, 1], s2[:, 2], s=42, color="#fb923c", alpha=0.30, depthshade=True)
            # Coarse-grained COM beads
            ax3d_r.scatter([c1c[0]], [c1c[1]], [c1c[2]], s=260, color="#38bdf8", edgecolors="#ffffff", linewidths=1.2)
            ax3d_r.scatter([c2c[0]], [c2c[1]], [c2c[2]], s=260, color="#fb923c", edgecolors="#ffffff", linewidths=1.2)
            # Orientation normal arrows
            ax3d_r.quiver(c1c[0], c1c[1], c1c[2], n1[0], n1[1], n1[2], color="#e0f2fe", lw=2.2, arrow_length_ratio=0.28)
            ax3d_r.quiver(c2c[0], c2c[1], c2c[2], n2[0], n2[1], n2[2], color="#ffedd5", lw=2.2, arrow_length_ratio=0.28)
            # Effective CG pair interaction line
            ax3d_r.plot([c1c[0], c2c[0]], [c1c[1], c2c[1]], [c1c[2], c2c[2]], "--", color="#22c55e", lw=1.4, alpha=0.75)

            # Update bottom cursors
            t_cur = t_axis[f]
            cursor_sep.set_xdata([t_cur, t_cur])
            dot_sep_t.set_data([t_cur], [r_true_np[f]])
            dot_sep_c.set_data([t_cur], [r_cg_np[f]])
            cursor_ang.set_xdata([t_cur, t_cur])
            dot_ang_t.set_data([t_cur], [L_true_np[f]])
            dot_ang_c.set_data([t_cur], [L_cg_np[f]])

            err_r = abs(r_cg_np[f] - r_true_np[f])
            title_txt.set_text(
                f"Held-Out Test Collision  |  t = {t_cur:.3f} τ   |   r_LAMMPS = {r_true_np[f]:.3f} σ   |   r_CG = {r_cg_np[f]:.3f} σ   (Δr = {err_r:.4f} σ)"
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved video: {OUT_MP4} ({OUT_MP4.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
