# φTD Ablation — Breakout & SpaceInvaders

Brief plan for the MinAtar φTD hyperparameter ablation (5 vmapped seeds; 10 seeds risk OOM).

## Axes

| # | Hyperparameter | Values |
|---|----------------|--------|
| 1 | Frequency sampling | Pareto (`pareto_1`), Exponential (`half_laplacian`), Uniform |
| 2 | `#` sampled frequencies (`NUM_OMEGA_SAMPLES`) | 32, 64, 128, 256 |
| 3 | `#` distributional components (`M_PARTICLES`) | 10, 20, 51, 100 |
| 4 | Distribution family | Gaussian (MoG), Categorical, Quantile |

Envs: `Breakout-MinAtar`, `SpaceInvaders-MinAtar`.

## Protocol (2 phases → 34 runs)

**Phase 1 — Gaussian only, one-factor sweeps (16 runs)**  
Hold sampling = Pareto. Sweep (2) then (3) independently (defaults for the held factor: `NUM_OMEGA_SAMPLES=128`, `M_PARTICLES=51`):

- 4 × 2 env = 8 frequency-count runs  
- 4 × 2 env = 8 component-count runs  

Pick best `NUM_OMEGA_SAMPLES` and best `M_PARTICLES` per env (or shared if tied).

**Phase 2 — sampling × family grid (18 runs)**  
Fix Phase-1 winners. Run all pairs of (1)×(4):

- 3 schemes × 3 families × 2 env = 18  

**Total:** \(16 + 18 = 34\) Slurm tasks.

## Slurm

```bash
# Phase 1 (16 tasks)
sbatch slurm/slurm_minatar_phi_td_ablation_phase1.sh

# Phase 2 — winners from Phase 1: N_ω=128, m=51 (18 tasks)
BEST_NUM_OMEGA=128 BEST_M_PARTICLES=51 \
  sbatch slurm/slurm_minatar_phi_td_ablation_phase2.sh
```

Phase-1 pick (kept for both envs): **`NUM_OMEGA_SAMPLES=128`**, **`M_PARTICLES=51`**.

## Plots

Uses the same final-report MinAtar multi-env style as `plot_minatar_10m_phi_td_mog_gamma_laplace_logistic`
(short env titles, legend on first panel, large fonts). Helpers live in `plot_wandb_minatar.py`.

```python
from plot_wandb_minatar import (
    plot_minatar_phi_td_ablation_phase1,
    plot_minatar_phi_td_ablation_phase2,
)

plot_minatar_phi_td_ablation_phase1(project="Deep-CVI-Experiments", entity="fatty_data")
plot_minatar_phi_td_ablation_phase2(project="Deep-CVI-Experiments", entity="fatty_data")
```

Outputs:

- `figures/minatar_phi_td_ablation_phase1_omega_count.png` (+ `.pdf`)
- `figures/minatar_phi_td_ablation_phase1_components.png` (+ `.pdf`)
- `figures/minatar_phi_td_ablation_phase2.png` (+ `.pdf`)

### Sparse CSVs (rebuttal tables)

Each plot also writes a long CSV with columns `env,variant,step,return`:

- checkpoints every **2M** steps up to 10M (2M, 4M, 6M, 8M, 10M)
- seed-mean episodic return with **one decimal place** (e.g. ``98.7``)

- `figures/minatar_phi_td_ablation_phase1_omega_count.csv`
- `figures/minatar_phi_td_ablation_phase1_components.csv`
- `figures/minatar_phi_td_ablation_phase2.csv`
