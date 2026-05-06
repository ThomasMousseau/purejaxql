## Build uv env

```bash
uv sync
```

## Weights & Biases (optional)

To log curves to W&B, edit `purejaxql/config/config.yaml`:

- Set your W&B workspace info:
  - `ENTITY: "your-entity"`
  - `PROJECT: "your-project"`
- Enable logging:
  - Set `WANDB_MODE: "online"` in `purejaxql/config/config.yaml`
  - If an algorithm YAML also defines `WANDB_MODE`, set it to `online` there too

By default, this submission is configured for offline-safe runs (`WANDB_MODE: disabled`).

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

