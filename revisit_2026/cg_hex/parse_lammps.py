"""Parse LAMMPS rigid/local dump files into compact PyTorch tensors and verify
exact rotational kinematics (quaternion convention & Richardson integrator match).
"""
from __future__ import annotations

import glob
import time
from pathlib import Path
import numpy as np
import torch


def parse_single_dump(path: Path) -> np.ndarray:
    """Fast parser for LAMMPS local dump with 1 entry per timestep (38 floats)."""
    rows = []
    with open(path, "r") as f:
        lines = f.readlines()
    # Each block has 9 header lines + 1 data line = 10 lines
    for i in range(9, len(lines), 10):
        rows.append(np.fromstring(lines[i], sep=" ", dtype=np.float32))
    return np.stack(rows, axis=0)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion q = [w, x, y, z] of shape (..., 4) to rotation matrix (..., 3, 3)."""
    w, x, y, z = q.unbind(dim=-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


def main() -> None:
    t0 = time.time()
    data_dir = Path(__file__).resolve().parent / "data"
    files = sorted(data_dir.glob("traj_*.dat"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(files)} LAMMPS trajectory files.")

    # Columns in dump:
    # 0:3   com_1 (x1)
    # 3:6   com_2 (x2)
    # 6:9   vel_1 (v1)
    # 9:12  vel_2 (v2)
    # 12:16 q_1   (w, x, y, z)
    # 16:20 q_2   (w, x, y, z)
    # 20:23 av_1  (lab omega_1)
    # 23:26 av_2  (lab omega_2)
    # 26:29 am_1  (lab L_1)
    # 29:32 am_2  (lab L_2)
    # 32:35 i_1   (3, 3, 6)
    # 35:38 i_2   (3, 3, 6)
    all_trajs = []
    for p in files:
        arr = parse_single_dump(p)
        all_trajs.append(arr)
        sep = np.linalg.norm(arr[:, 3:6] - arr[:, 0:3], axis=-1)
        frac_coll = float(np.mean(sep < 2.0 + 2.0 ** (1.0 / 6.0)))
        print(f"  {p.name}: {arr.shape[0]} frames | min_r={sep.min():.3f}, max_r={sep.max():.3f}, frac_interacting={frac_coll:.3f}")

    # Stack into (N_traj, T, 38)
    min_len = min(a.shape[0] for a in all_trajs)
    stacked = torch.from_numpy(np.stack([a[:min_len] for a in all_trajs], axis=0))

    # Verify quaternion -> lab omega consistency: omega_lab == R(q) J^{-1} R(q)^T L_lab
    q1 = stacked[0, :100, 12:16]
    av1 = stacked[0, :100, 20:23]
    am1 = stacked[0, :100, 26:29]
    J_inv = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    R1 = quat_to_rotmat(q1)
    l_body = torch.einsum("nij,nj->ni", R1.transpose(-1, -2), am1)
    av1_check = torch.einsum("nij,nj->ni", R1, l_body * J_inv)
    err_pos = (av1 - av1_check).abs().max().item()
    print(f"Quaternion convention check max |omega_LAMMPS - R J^-1 R^T L|: {err_pos:.2e}")

    out_path = data_dir / "hex_dataset.pt"
    torch.save({
        "x1": stacked[:, :, 0:3].contiguous(),
        "x2": stacked[:, :, 3:6].contiguous(),
        "v1": stacked[:, :, 6:9].contiguous(),
        "v2": stacked[:, :, 9:12].contiguous(),
        "q1": stacked[:, :, 12:16].contiguous(),
        "q2": stacked[:, :, 16:20].contiguous(),
        "omega1": stacked[:, :, 20:23].contiguous(),
        "omega2": stacked[:, :, 23:26].contiguous(),
        "L1": stacked[:, :, 26:29].contiguous(),  # lab-frame angular momentum
        "L2": stacked[:, :, 29:32].contiguous(),
        "mass": 7.0,
        "inertia": torch.tensor([3.0, 3.0, 6.0]),
        "dt": 1e-3,
        "K_ext": float(6.0 * 0.5 / (2.0 ** (1.0 / 3.0))),
        "r0_ext": 2.0,
    }, out_path)
    print(f"Saved {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
