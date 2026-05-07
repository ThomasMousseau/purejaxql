## Build uv env

```bash
uv sync
```

## Weights & Biases

Training entry points call `wandb.init` with `ENTITY`, `PROJECT`, and `WANDB_MODE` from config.

**You must edit `purejaxql/config/config.yaml`** before running with the default `WANDB_MODE: "online"`:

- Set `ENTITY` to your W&B username or team (not the placeholder `your-entity`).
- Set `PROJECT` to an existing or new project name (not `your-project`).
- Run `wandb login` on the machine if you have not already.

Leaving the placeholders while `WANDB_MODE` is `online` will fail at startup (typically HTTP 404 from W&B). Algorithm YAMLs under `config/alg/` set `WANDB_MODE: online` so logging stays on; override with `WANDB_MODE=disabled` (Hydra) or change the base config if you want fully local runs without W&B.

## Distribution analysis

Run from YAML config:

```bash
uv run python run_distribution_analysis.py
```

Replot latest run:

```bash
uv run python run_distribution_analysis.py --replot
```

## MinAtar

Launch format (Python file + YAML config override):

```bash
uv run python -m purejaxql.<python_file_without_.py> +alg=<yaml_config_name_without_.yaml>
```

Examples:

```bash
uv run python -m purejaxql.pqn_minatar +alg=pqn_minatar
uv run python -m purejaxql.ctd_minatar +alg=ctd_minatar
uv run python -m purejaxql.qtd_minatar +alg=qtd_minatar
uv run python -m purejaxql.phi_td_pqn_minatar +alg=phi_td_minatar_mog
uv run python -m purejaxql.phi_td_pqn_minatar +alg=phi_td_minatar_categorical
uv run python -m purejaxql.phi_td_pqn_minatar +alg=phi_td_minatar_quantile
uv run python -m purejaxql.phi_td_pqn_minatar +alg=phi_td_minatar_dirac
uv run python -m purejaxql.phi_td_pqn_minatar +alg=phi_td_minatar_cauchy
uv run python -m purejaxql.phi_td_pqn_minatar +alg=phi_td_minatar_gamma
uv run python -m purejaxql.phi_td_pqn_minatar +alg=phi_td_minatar_laplace
uv run python -m purejaxql.phi_td_pqn_minatar +alg=phi_td_minatar_logistic
```

## Craftax

Launch format (Python file + YAML config override):

```bash
uv run python -m purejaxql.<python_file_without_.py> +alg=<yaml_config_name_without_.yaml>
```

Examples:

```bash
uv run python -m purejaxql.pqn_rnn_craftax +alg=pqn_rnn_craftax
uv run python -m purejaxql.ctd_rnn_craftax +alg=ctd_rnn_craftax
uv run python -m purejaxql.qtd_rnn_craftax +alg=qtd_rnn_craftax
uv run python -m purejaxql.phi_td_pqn_rnn_craftax +alg=phi_td_rnn_craftax_mog
uv run python -m purejaxql.phi_td_pqn_rnn_craftax +alg=phi_td_rnn_craftax_categorical
uv run python -m purejaxql.phi_td_pqn_rnn_craftax +alg=phi_td_rnn_craftax_quantile
uv run python -m purejaxql.phi_td_pqn_rnn_craftax +alg=phi_td_rnn_craftax_dirac
uv run python -m purejaxql.phi_td_pqn_rnn_craftax +alg=phi_td_rnn_craftax_cauchy
uv run python -m purejaxql.phi_td_pqn_rnn_craftax +alg=phi_td_rnn_craftax_gamma
uv run python -m purejaxql.phi_td_pqn_rnn_craftax +alg=phi_td_rnn_craftax_laplace
uv run python -m purejaxql.phi_td_pqn_rnn_craftax +alg=phi_td_rnn_craftax_logistic
```

## Naming note (Half-Laplace vs Exponential)

In this project, a Half-Laplace distribution with parameters `(mu, b)` (with `mu = 0`) is equivalent to an Exponential distribution with rate `lambda = 1 / b`.
Some code paths still use names like `Half-Laplacian` or `Half-Laplace` due to older experiments. We originally wanted a distribution spanning both `R+` and `R-`, then used conjugate symmetry of the characteristic function instead, so we no longer needed to sample from both sides. The naming stayed for backward compatibility.

