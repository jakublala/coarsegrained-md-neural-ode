"""Velocity-Verlet integrator and batched Hamiltonian Monte Carlo (HMC) / Neural HMC."""
from __future__ import annotations

from typing import Callable
import torch


def velocity_verlet_step(
    x: torch.Tensor,
    p: torch.Tensor,
    f: torch.Tensor,
    force_fn: Callable[[torch.Tensor], torch.Tensor],
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single leapfrog / Velocity-Verlet step (Eqs. 16-18, m = 1)."""
    p_half = p + 0.5 * dt * f
    x_new = x + dt * p_half
    f_new = force_fn(x_new)
    p_new = p_half + 0.5 * dt * f_new
    return x_new, p_new, f_new


def integrate_verlet(
    x0: torch.Tensor,
    p0: torch.Tensor,
    force_fn: Callable[[torch.Tensor], torch.Tensor],
    dt: float,
    n_steps: int,
    record: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate Hamiltonian dynamics for n_steps using Velocity-Verlet.

    If record=True, returns (X_traj, P_traj) of shape (..., n_steps + 1, dim).
    If record=False, returns final (x, p) of shape (..., dim).
    """
    x = x0
    p = p0
    f = force_fn(x)
    if record:
        xs = [x]
        ps = [p]
        for _ in range(n_steps):
            x, p, f = velocity_verlet_step(x, p, f, force_fn, dt)
            xs.append(x)
            ps.append(p)
        return torch.stack(xs, dim=-2), torch.stack(ps, dim=-2)
    else:
        for _ in range(n_steps):
            x, p, f = velocity_verlet_step(x, p, f, force_fn, dt)
        return x, p


@torch.no_grad()
def run_hmc_batched(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    proposal_force_fn: Callable[[torch.Tensor], torch.Tensor],
    x_init: torch.Tensor,
    dt: float,
    n_steps: int,
    n_samples: int,
    beta: float = 1.0,
    burn_in: int = 200,
    record_trajectories: bool = False,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Run batched HMC chains in parallel.

    Args:
        energy_fn: Potential energy used in Metropolis acceptance (Eq. 20).
                   For regular HMC and Neural HMC - Explicit: exact V(x).
                   For Neural HMC - Model: learned V'(x).
        proposal_force_fn: Force function used in Velocity-Verlet proposal.
        x_init: Initial positions of shape (n_chains, dim).
        dt: Integration time step.
        n_steps: Number of Velocity-Verlet steps per HMC proposal.
        n_samples: Number of MC samples to collect after burn-in.
        beta: Inverse temperature (1 / k_B T = 1).
        burn_in: Number of burn-in MC steps before recording.
        record_trajectories: If True, also store full proposal trajectories.
        seed: Random seed.

    Returns:
        Dictionary with 'samples' (n_chains, n_samples, dim), 'accept_rate',
        and optionally 'traj_x', 'traj_p', 'accepted'.
    """
    gen = torch.Generator(device=x_init.device)
    gen.manual_seed(seed)

    n_chains, dim = x_init.shape
    x = x_init.clone()
    v_curr = energy_fn(x)

    samples = torch.empty(n_chains, n_samples, dim, dtype=x.dtype, device=x.device)
    if record_trajectories:
        traj_x = torch.empty(n_chains, n_samples, n_steps + 1, dim, dtype=x.dtype, device=x.device)
        traj_p = torch.empty(n_chains, n_samples, n_steps + 1, dim, dtype=x.dtype, device=x.device)
        accepted_flags = torch.empty(n_chains, n_samples, dtype=torch.bool, device=x.device)

    total_iters = burn_in + n_samples
    accept_count = 0

    for it in range(total_iters):
        p0 = torch.randn(x.shape, generator=gen, dtype=x.dtype, device=x.device)
        k_curr = 0.5 * torch.sum(p0 ** 2, dim=-1)
        h_curr = v_curr + k_curr

        if record_trajectories and it >= burn_in:
            tx, tp = integrate_verlet(x, p0, proposal_force_fn, dt, n_steps, record=True)
            x_prop = tx[:, -1, :]
            p_prop = tp[:, -1, :]
        else:
            x_prop, p_prop = integrate_verlet(x, p0, proposal_force_fn, dt, n_steps, record=False)

        v_prop = energy_fn(x_prop)
        k_prop = 0.5 * torch.sum(p_prop ** 2, dim=-1)
        h_prop = v_prop + k_prop

        # Handle NaN/Inf proposals (reject them cleanly)
        delta_h = h_prop - h_curr
        valid = torch.isfinite(delta_h)
        log_u = torch.log(torch.rand(n_chains, generator=gen, dtype=x.dtype, device=x.device).clamp(min=1e-12))
        accept = valid & (log_u < (-beta * delta_h))

        x = torch.where(accept.unsqueeze(-1), x_prop, x)
        v_curr = torch.where(accept, v_prop, v_curr)

        if it >= burn_in:
            idx = it - burn_in
            samples[:, idx, :] = x
            accept_count += int(accept.sum().item())
            if record_trajectories:
                traj_x[:, idx] = tx
                traj_p[:, idx] = tp
                accepted_flags[:, idx] = accept

    out = {
        "samples": samples,
        "accept_rate": torch.tensor(accept_count / (n_chains * n_samples)),
    }
    if record_trajectories:
        out["traj_x"] = traj_x
        out["traj_p"] = traj_p
        out["accepted"] = accepted_flags
    return out


@torch.no_grad()
def generate_dataset(
    potential,
    n_trajectories: int,
    n_steps: int = 100,
    n_chains: int = 20,
    burn_in: int = 200,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Generate HMC proposal trajectories {X, P} of shape (n_trajectories, n_steps+1, dim)."""
    assert n_trajectories % n_chains == 0
    samples_per_chain = n_trajectories // n_chains
    gen = torch.Generator()
    gen.manual_seed(seed)
    x_init = potential.init_position(n_chains, generator=gen)

    res = run_hmc_batched(
        energy_fn=potential.energy,
        proposal_force_fn=potential.force,
        x_init=x_init,
        dt=potential.dt,
        n_steps=n_steps,
        n_samples=samples_per_chain,
        beta=1.0,
        burn_in=burn_in,
        record_trajectories=True,
        seed=seed + 1,
    )
    # Flatten (n_chains, samples_per_chain, n_steps+1, dim) -> (n_trajectories, n_steps+1, dim)
    X = res["traj_x"].reshape(n_trajectories, n_steps + 1, potential.dim)
    P = res["traj_p"].reshape(n_trajectories, n_steps + 1, potential.dim)
    return X, P, float(res["accept_rate"])
