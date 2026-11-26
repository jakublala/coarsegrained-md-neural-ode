"""Fixed held-out evaluation benchmark for the autoresearch hill-climb.

Single scalar metric D (Trajectory Deviation Score, lower is better, 0.0 = exact LAMMPS):
  Evaluated over 10 distinct held-out collision windows (200 steps = 0.2 tau each) from traj_10.
  D = 0.25 * (MAE(dx)/s_dx + MAE(dv)/s_dv + MAE(L)/s_L + MeanAngleErr_deg / 10.0)
  where s_dx, s_dv, s_L are the ground-truth change scales across those 200-step collisions
  (so a trivial V_CG = 0 model scores ~1.0, and exact_hex_potential scores ~0.0005).
"""
from __future__ import annotations

import math
from pathlib import Path
import torch
import torch.nn as nn

from rigid_ode import exact_hex_potential, rollout_rigid_body, R_MAX

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "hex_dataset.pt"


def build_benchmark(d: dict | None = None, n_windows: int = 10, n_steps: int = 200) -> dict:
    if d is None:
        d = torch.load(DATA_PATH, weights_only=True)
    # Test trajectory is index 9 (traj_10, never seen during training)
    dx = d["x2"][9] - d["x1"][9]
    dv = d["v2"][9] - d["v1"][9]
    q1, q2 = d["q1"][9], d["q2"][9]
    L1, L2 = d["L1"][9], d["L2"][9]

    # Detect actual collision peaks via angular momentum impulse |dL1| + |dL2|
    dL = (L1[1:] - L1[:-1]).norm(dim=-1) + (L2[1:] - L2[:-1]).norm(dim=-1)
    # Pick 10 well-separated collision events spaced >= 4000 frames apart
    starts = []
    t = 1000
    while len(starts) < n_windows and t < len(dL) - n_steps - 100:
        if dL[t] > 5e-3:
            # Start 30 steps before the impulse so the rollout enters, impacts, and exits cleanly
            starts.append(max(0, t - 30))
            t += 4000
        else:
            t += 1

    t_off = torch.arange(n_steps + 1).unsqueeze(0)
    t_mat = torch.tensor(starts, dtype=torch.long).unsqueeze(1) + t_off  # (W, T+1)

    bench = {
        "starts": starts,
        "n_steps": n_steps,
        "dx": dx[t_mat],
        "dv": dv[t_mat],
        "q1": q1[t_mat],
        "q2": q2[t_mat],
        "L1": L1[t_mat],
        "L2": L2[t_mat],
        "dt": d["dt"],
        "K_ext": d["K_ext"],
        "r0_ext": d["r0_ext"],
        "mass": d["mass"],
    }

    # Compute reference scales from the zero-potential (V_CG = 0) baseline so D(V=0) == 1.000!
    zero_fn = lambda dx_i, q1_i, q2_i: torch.zeros(dx_i.shape[0], device=dx_i.device, dtype=dx_i.dtype)
    with torch.enable_grad():
        p0 = rollout_rigid_body(
            zero_fn,
            bench["dx"][:, 0], bench["dv"][:, 0],
            bench["q1"][:, 0], bench["q2"][:, 0],
            bench["L1"][:, 0], bench["L2"][:, 0],
            dt=bench["dt"], n_steps=n_steps,
            K_ext=bench["K_ext"], r0_ext=bench["r0_ext"], mass=bench["mass"],
            create_graph=False, record_all=True,
        )
    bench["norm_dx"] = float((p0["dx"] - bench["dx"]).abs().mean().item())
    bench["norm_dv"] = float((p0["dv"] - bench["dv"]).abs().mean().item())
    bench["norm_L"] = float(0.5 * ((p0["L1"] - bench["L1"]).abs().mean() + (p0["L2"] - bench["L2"]).abs().mean()).item())
    dot0 = 0.5 * (
        (p0["q1"].double() * bench["q1"].double()).sum(dim=-1).abs().clamp(0.0, 1.0)
        + (p0["q2"].double() * bench["q2"].double()).sum(dim=-1).abs().clamp(0.0, 1.0)
    )
    bench["norm_deg"] = float((2.0 * torch.acos(dot0) * (180.0 / math.pi)).mean().item())
    return bench


def score_model(pot_fn, bench: dict) -> dict:
    """Compute the single scalar Trajectory Deviation Score D (0.0 = exact, 1.0 = V_CG=0)."""
    if isinstance(pot_fn, nn.Module):
        pot_fn.eval()
    with torch.enable_grad():
        p = rollout_rigid_body(
            pot_fn,
            bench["dx"][:, 0], bench["dv"][:, 0],
            bench["q1"][:, 0], bench["q2"][:, 0],
            bench["L1"][:, 0], bench["L2"][:, 0],
            dt=bench["dt"], n_steps=bench["n_steps"],
            K_ext=bench["K_ext"], r0_ext=bench["r0_ext"], mass=bench["mass"],
            create_graph=False, record_all=True,
        )
    dx_mae = float((p["dx"] - bench["dx"]).abs().mean().item())
    dv_mae = float((p["dv"] - bench["dv"]).abs().mean().item())
    L_mae = float(0.5 * ((p["L1"] - bench["L1"]).abs().mean() + (p["L2"] - bench["L2"]).abs().mean()).item())
    dot1 = (p["q1"].double() * bench["q1"].double()).sum(dim=-1).abs().clamp(0.0, 1.0)
    dot2 = (p["q2"].double() * bench["q2"].double()).sum(dim=-1).abs().clamp(0.0, 1.0)
    deg_mae = float(0.5 * (
        (2.0 * torch.acos(dot1) * (180.0 / math.pi)).mean()
        + (2.0 * torch.acos(dot2) * (180.0 / math.pi)).mean()
    ).item())

    D = 0.25 * (
        dx_mae / bench["norm_dx"]
        + dv_mae / bench["norm_dv"]
        + L_mae / bench["norm_L"]
        + deg_mae / bench["norm_deg"]
    )
    return {
        "D": D,
        "dx_mae": dx_mae,
        "dv_mae": dv_mae,
        "L_mae": L_mae,
        "deg_mae": deg_mae,
    }


if __name__ == "__main__":
    bench = build_benchmark()
    exact_res = score_model(exact_hex_potential, bench)
    print(f"Benchmark built ({len(bench['starts'])} collisions x {bench['n_steps']} steps)")
    print(f"  Zero-potential baseline D = 1.0000 (norm_dx={bench['norm_dx']:.4f}, norm_dv={bench['norm_dv']:.4f}, norm_L={bench['norm_L']:.4f}, norm_deg={bench['norm_deg']:.2f}°)")
    print(f"  Exact 7-bead LJ floor   D = {exact_res['D']:.6f} (deg_mae={exact_res['deg_mae']:.4f}°)")
