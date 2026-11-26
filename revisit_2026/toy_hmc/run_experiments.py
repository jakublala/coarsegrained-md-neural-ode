"""Master runner to reproduce Chapter 4 (Toy Potentials & Neural HMC) of the thesis:
- Generates datasets (20,000 train + 5,000 test trajectories) for 2D Shell, 10D Gaussian, 2D Wolfe-Quapp
- Produces Figure 7 and Figure 9
- Runs Table 1 (loss function comparison across 5 seeds)
- Runs Learning Curves (7 dataset sizes x 5 seeds per potential)
- Trains 5 full-dataset models per potential & runs 25x HMC, Neural HMC-Explicit, Neural HMC-Model
- Produces Figure 10a, 10b, 10c and saves all numerical results to results/summary.json
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from potentials import POTENTIALS
from hmc import generate_dataset, run_hmc_batched
from neural_ode import PotentialNet, train_model, rollout_neural_ode, compute_loss
from plot_figures import plot_fig7_hmc_shell, plot_fig9_toy_potentials, plot_fig10_potential


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

LC_SIZES = [100, 200, 500, 1000, 2000, 5000, 10000]
HMC_SAMPLE_COUNTS = [100, 200, 500, 1000, 2000, 5000, 10000]
N_SEEDS = 5
MAX_WORKERS = min(10, os.cpu_count() or 4)


def _train_worker(args: dict) -> dict:
    """Worker function for parallel model training (1 thread per process)."""
    torch.set_num_threads(1)
    data = torch.load(args["dataset_path"], weights_only=True)
    X_tr = data["X_train"][: args["n_train"]]
    P_tr = data["P_train"][: args["n_train"]]
    X_te = data["X_test"]
    P_te = data["P_test"]

    res = train_model(
        X_train=X_tr,
        P_train=P_tr,
        X_test=X_te,
        P_test=P_te,
        dt=args["dt"],
        init_lr=args["init_lr"],
        epochs=args["epochs"],
        batch_size=800,
        rollout_steps=10,
        loss_type=args["loss_type"],
        sched_freq=args["sched_freq"],
        sched_factor=0.6,
        seed=args["seed"],
    )

    # Also compute 100-step test loss for completeness
    model = res["model"]
    with torch.no_grad():
        xh100, ph100 = rollout_neural_ode(model, X_te[:, 0], P_te[:, 0], args["dt"], 100)
        loss100 = float(compute_loss(X_te, xh100, P_te, ph100, loss_type="both").item())

    out = {
        "tag": args["tag"],
        "pot_key": args["pot_key"],
        "loss_type": args["loss_type"],
        "n_train": args["n_train"],
        "seed": args["seed"],
        "test_loss_10": res["test_loss"],
        "test_loss_100": loss100,
    }
    if args.get("save_model_path"):
        torch.save(model.state_dict(), args["save_model_path"])
        out["model_path"] = args["save_model_path"]
    return out


def _hmc_worker(args: dict) -> dict:
    """Worker function for running batched HMC / Neural HMC chains."""
    torch.set_num_threads(1)
    pot = POTENTIALS[args["pot_key"]]()
    mode = args["mode"]  # 'hmc', 'neural_explicit', 'neural_model'
    n_chains = args["n_chains"]
    n_samples = max(HMC_SAMPLE_COUNTS)

    gen = torch.Generator().manual_seed(args["seed"])
    x_init = pot.init_position(n_chains, generator=gen)

    if mode == "hmc":
        energy_fn = pot.energy
        force_fn = pot.force
    else:
        model = PotentialNet(dim=pot.dim, hidden_dim=50)
        model.load_state_dict(torch.load(args["model_path"], weights_only=True))
        model.eval()
        force_fn = model.force
        energy_fn = pot.energy if mode == "neural_explicit" else model.energy

    res = run_hmc_batched(
        energy_fn=energy_fn,
        proposal_force_fn=force_fn,
        x_init=x_init,
        dt=pot.dt,
        n_steps=100,
        n_samples=n_samples,
        beta=1.0,
        burn_in=200,
        record_trajectories=False,
        seed=args["seed"] + 77,
    )
    samples = res["samples"]  # (n_chains, n_samples, dim)
    # Compute |<x1>| and |<x2>| for each chain at each sample count
    abs_means_x1 = {}
    abs_means_x2 = {}
    for n in HMC_SAMPLE_COUNTS:
        sub = samples[:, :n, :]
        abs_means_x1[n] = torch.abs(sub[:, :, 0].mean(dim=1)).tolist()
        abs_means_x2[n] = torch.abs(sub[:, :, 1].mean(dim=1)).tolist()

    return {
        "pot_key": args["pot_key"],
        "mode": mode,
        "accept_rate": float(res["accept_rate"]),
        "abs_means_x1": abs_means_x1,
        "abs_means_x2": abs_means_x2,
    }


def main() -> None:
    t_start = time.time()
    print(f"Running Chapter 4 reproduction with {MAX_WORKERS} parallel workers...")

    # 1. Generate static figures (Fig 7 & Fig 9)
    print("\n[1/5] Generating Figure 7 and Figure 9...")
    plot_fig7_hmc_shell(FIGURES_DIR / "fig7_hmc_shell.png")
    plot_fig9_toy_potentials(FIGURES_DIR / "fig9_toy_potentials.png")
    print("  Saved fig7_hmc_shell.png and fig9_toy_potentials.png")

    # 2. Generate datasets for all 3 potentials
    print("\n[2/5] Generating HMC datasets (20,000 train + 5,000 test trajectories each)...")
    dataset_paths = {}
    for idx, (pot_key, pot_cls) in enumerate(POTENTIALS.items()):
        pot = pot_cls()
        dpath = RESULTS_DIR / f"dataset_{pot_key}.pt"
        dataset_paths[pot_key] = str(dpath)
        if not dpath.exists():
            t0 = time.time()
            X_tr, P_tr, acc_tr = generate_dataset(pot, 20000, n_steps=100, n_chains=20, burn_in=200, seed=100 + idx * 10)
            X_te, P_te, acc_te = generate_dataset(pot, 5000, n_steps=100, n_chains=20, burn_in=200, seed=200 + idx * 10)
            torch.save({"X_train": X_tr, "P_train": P_tr, "X_test": X_te, "P_test": P_te}, dpath)
            print(f"  {pot.name}: generated in {time.time() - t0:.1f}s (accept_rate={acc_tr:.3f})")
        else:
            print(f"  {pot.name}: loaded cached {dpath.name}")

    # 3. Build all training jobs:
    #    - Table 1: 3 potentials x 3 loss types (x, p, both) x 5 seeds on full 20,000 dataset (4,000 epochs)
    #      (save the 'both' models for Neural HMC & contour/profile plots!)
    #    - Learning curves: 3 potentials x 7 sizes x 5 seeds (4,000 epochs)
    print("\n[3/5] Launching parallel Neural ODE training jobs (150 models total)...")
    jobs = []
    for pot_key, pot_cls in POTENTIALS.items():
        pot = pot_cls()
        # Table 1 + Full models for Neural HMC
        for loss_type in ("x", "p", "both"):
            for s in range(N_SEEDS):
                save_path = None
                epochs = 4000
                sched_freq = 1000
                if loss_type == "both":
                    save_path = str(RESULTS_DIR / f"model_{pot_key}_seed{s}.pt")
                    epochs = 5000
                    sched_freq = 1200
                jobs.append({
                    "tag": "table1",
                    "pot_key": pot_key,
                    "dataset_path": dataset_paths[pot_key],
                    "dt": pot.dt,
                    "init_lr": pot.init_lr,
                    "n_train": 20000,
                    "loss_type": loss_type,
                    "epochs": epochs,
                    "sched_freq": sched_freq,
                    "seed": 1000 + s,
                    "save_model_path": save_path,
                })
        # Learning curves
        for n_tr in LC_SIZES:
            for s in range(N_SEEDS):
                jobs.append({
                    "tag": "lc",
                    "pot_key": pot_key,
                    "dataset_path": dataset_paths[pot_key],
                    "dt": pot.dt,
                    "init_lr": pot.init_lr,
                    "n_train": n_tr,
                    "loss_type": "both",
                    "epochs": 4000,
                    "sched_freq": 1000,
                    "seed": 2000 + s,
                })

    t0 = time.time()
    table1_raw = {k: {lt: [] for lt in ("x", "p", "both")} for k in POTENTIALS}
    table1_raw_100 = {k: {lt: [] for lt in ("x", "p", "both")} for k in POTENTIALS}
    lc_raw = {k: {n: [] for n in LC_SIZES} for k in POTENTIALS}
    full_models = {k: [] for k in POTENTIALS}

    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_train_worker, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r["tag"] == "table1":
                table1_raw[r["pot_key"]][r["loss_type"]].append(r["test_loss_10"])
                table1_raw_100[r["pot_key"]][r["loss_type"]].append(r["test_loss_100"])
                if r["loss_type"] == "both":
                    full_models[r["pot_key"]].append((r["test_loss_10"], r["model_path"], r["seed"]))
            else:
                lc_raw[r["pot_key"]][r["n_train"]].append(r["test_loss_10"])
            if done % 25 == 0 or done == len(jobs):
                print(f"  Completed {done}/{len(jobs)} trainings ({time.time() - t0:.1f}s elapsed)")

    # Print Table 1
    print("\n=== Table 1: Test Loss (10-step MAE) ===")
    table1_summary = {}
    for lt in ("x", "p", "both"):
        row = {}
        line = f"  Loss={lt:4s} | "
        for pot_key in ("shell", "gaussian", "wolfe_quapp"):
            vals = np.array(table1_raw[pot_key][lt])
            vals100 = np.array(table1_raw_100[pot_key][lt])
            row[pot_key] = {
                "mean_10": float(vals.mean()),
                "std_10": float(vals.std(ddof=1)),
                "mean_100": float(vals100.mean()),
                "std_100": float(vals100.std(ddof=1)),
            }
            line += f"{pot_key}: {vals.mean():.5f} ± {vals.std(ddof=1):.5f}  | "
        table1_summary[lt] = row
        print(line)

    # 4. Run HMC, Neural HMC - Explicit, and Neural HMC - Model (25 runs each per potential)
    print("\n[4/5] Running HMC and Neural HMC simulations (25 chains x 10,000 MC samples each)...")
    t0 = time.time()
    hmc_jobs = []
    for pot_key in POTENTIALS:
        # Regular HMC: 25 chains
        hmc_jobs.append({
            "pot_key": pot_key,
            "mode": "hmc",
            "n_chains": 25,
            "seed": 5000,
        })
        # Neural HMC (Explicit & Model): 5 models x 5 chains = 25 chains each
        full_models[pot_key].sort(key=lambda t: t[0])
        for m_idx, (_, m_path, _) in enumerate(full_models[pot_key]):
            for mode in ("neural_explicit", "neural_model"):
                hmc_jobs.append({
                    "pot_key": pot_key,
                    "mode": mode,
                    "model_path": m_path,
                    "n_chains": 5,
                    "seed": 6000 + m_idx * 100 + (0 if mode == "neural_explicit" else 50),
                })

    hmc_collect = {
        k: {
            m: {"x1": {n: [] for n in HMC_SAMPLE_COUNTS}, "x2": {n: [] for n in HMC_SAMPLE_COUNTS}, "acc": []}
            for m in ("hmc", "neural_explicit", "neural_model")
        }
        for k in POTENTIALS
    }

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_hmc_worker, j) for j in hmc_jobs]
        for fut in as_completed(futs):
            r = fut.result()
            slot = hmc_collect[r["pot_key"]][r["mode"]]
            slot["acc"].append(r["accept_rate"])
            for n in HMC_SAMPLE_COUNTS:
                slot["x1"][n].extend(r["abs_means_x1"][n])
                slot["x2"][n].extend(r["abs_means_x2"][n])
    print(f"  HMC simulations finished in {time.time() - t0:.1f}s")

    # 5. Aggregate stats & plot Figure 10a, 10b, 10c
    print("\n[5/5] Generating Figure 10a, 10b, 10c and saving summary...")
    summary = {"table1": table1_summary, "potentials": {}}
    fig_names = {
        "shell": "fig10a_shell.png",
        "gaussian": "fig10b_gaussian.png",
        "wolfe_quapp": "fig10c_wolfe_quapp.png",
    }

    for pot_key, pot_cls in POTENTIALS.items():
        pot = pot_cls()
        lc_means = [float(np.mean(lc_raw[pot_key][n])) for n in LC_SIZES]
        lc_stds = [float(np.std(lc_raw[pot_key][n], ddof=1)) for n in LC_SIZES]

        # Load best model for contour & side-profile visualization
        best_loss, best_model_path, _ = full_models[pot_key][0]
        rep_model = PotentialNet(dim=pot.dim, hidden_dim=50)
        rep_model.load_state_dict(torch.load(best_model_path, weights_only=True))
        rep_model.eval()

        hmc_stats = {}
        for mode in ("hmc", "neural_explicit", "neural_model"):
            slot = hmc_collect[pot_key][mode]
            hmc_stats[mode] = {
                "accept_rate": float(np.mean(slot["acc"])),
                "x1_mean": [float(np.mean(slot["x1"][n])) for n in HMC_SAMPLE_COUNTS],
                "x1_std": [float(np.std(slot["x1"][n], ddof=1)) for n in HMC_SAMPLE_COUNTS],
                "x2_mean": [float(np.mean(slot["x2"][n])) for n in HMC_SAMPLE_COUNTS],
                "x2_std": [float(np.std(slot["x2"][n], ddof=1)) for n in HMC_SAMPLE_COUNTS],
            }

        plot_fig10_potential(
            pot_key=pot_key,
            pot=pot,
            lc_sizes=LC_SIZES,
            lc_means=lc_means,
            lc_stds=lc_stds,
            rep_model=rep_model,
            hmc_sample_counts=HMC_SAMPLE_COUNTS,
            hmc_stats=hmc_stats,
            out_path=FIGURES_DIR / fig_names[pot_key],
        )
        print(f"  Saved {fig_names[pot_key]} (best_loss={best_loss:.5f}, acc_hmc={hmc_stats['hmc']['accept_rate']:.3f}, acc_exp={hmc_stats['neural_explicit']['accept_rate']:.3f}, acc_mod={hmc_stats['neural_model']['accept_rate']:.3f})")

        summary["potentials"][pot_key] = {
            "lc_sizes": LC_SIZES,
            "lc_means": lc_means,
            "lc_stds": lc_stds,
            "hmc_sample_counts": HMC_SAMPLE_COUNTS,
            "hmc_stats": hmc_stats,
        }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll done in {time.time() - t_start:.1f}s!")


if __name__ == "__main__":
    main()
