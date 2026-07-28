"""Shared plotting palette for distributional algorithm comparisons (paper canon)."""

from __future__ import annotations

# --- Base families (cross-figure consistency) ---
COLOR_MED_BLUE = "#008dff"  # IQN (legacy benchmarks)
COLOR_PINK_CTD = "#ff73b6"  # CTD (light pink)
COLOR_PURPLE_QTD = "#c701ff"  # QTD (magenta-purple)
COLOR_GREEN_MOG = "#4ecb8d"  # MoG / MoG-CQN
COLOR_DARK_BLUE_PQN = "#003a7d"  # PQN

# φTD Dirac / Categorical / Quantile — contrast with CTD pink & QTD purple on the same figure
COLOR_PHITD_FM = "#a855f7"  # purple (Dirac / F_m)
COLOR_PHITD_FCM = "#ea580c"  # orange (Categorical / F_{C,m})
COLOR_PHITD_FQM = "#16a34a"  # green (Quantile / F_{Q,m})

# φTD mixture CF variants (distinct, stable across experiments)
COLOR_PHITD_MOG =  "#ff0000" #d97706"  # Gaussian / MoG
COLOR_PHITD_MOGAMMA = "#f59e0b"
COLOR_PHITD_LAPLACE = "#008dff"
COLOR_PHITD_LOGISTIC = "#4e4e4c" # grey 
COLOR_PHITD_CAUCHY = "#92400e"

# --- Exact W&B algo tags → hex (canonical paper colors) ---
_WANDB_TAG_HEX: dict[str, str] = {
    # Core
    "MoG": COLOR_GREEN_MOG,
    "PQN": COLOR_DARK_BLUE_PQN,
    "CTD": COLOR_PINK_CTD,
    "QTD": COLOR_PURPLE_QTD,
    "IQN": COLOR_MED_BLUE,
    "CQN": COLOR_GREEN_MOG,
    "QR-DQN": COLOR_PURPLE_QTD,
    "C51": COLOR_PINK_CTD,
    # RNN Craftax tags
    "MOG_PQN_RNN": COLOR_GREEN_MOG,
    "PQN_RNN": COLOR_DARK_BLUE_PQN,
    "IQN_RNN": COLOR_MED_BLUE,
    "QTD_RNN": COLOR_PURPLE_QTD,
    "CTD_RNN": COLOR_PINK_CTD,
    # φTD MinAtar tags
    "PhiTD-Fm": COLOR_PHITD_FM,
    "PhiTD-FCm": COLOR_PHITD_FCM,
    "PhiTD-FQm": COLOR_PHITD_FQM,
    "PhiTD-Categorical": COLOR_PHITD_FCM,
    "PhiTD-Quantile": COLOR_PHITD_FQM,
    "PhiTD-Dirac": COLOR_PHITD_FM,
    "PhiTD-MoG": COLOR_PHITD_MOG,
    # MoG-PQN weighted-CF sweep (``plot_minatar_20m_td_lambda_aux_3exp`` / mimic vs baseline)
    "MoG-WeightedCF": "#ff0000",
    "PhiTD-MoGamma": COLOR_PHITD_MOGAMMA,
    "PhiTD-Laplace": COLOR_PHITD_LAPLACE,
    "PhiTD-Logistic": COLOR_PHITD_LOGISTIC,
    "PhiTD-Cauchy": COLOR_PHITD_CAUCHY,
    # Craftax RNN φTD — ``ALG_NAME.upper()`` from ``phi_td_*_rnn`` configs (same hues as PhiTD-*).
    "PHI_TD_DIRAC_RNN": COLOR_PHITD_FM,
    "PHI_TD_CATEGORICAL_RNN": COLOR_PHITD_FCM,
    "PHI_TD_QUANTILE_RNN": COLOR_PHITD_FQM,
    "PHI_TD_MOG_RNN": COLOR_PHITD_MOG,
    "PHI_TD_GAMMA_RNN": COLOR_PHITD_MOGAMMA,
    "PHI_TD_LAPLACE_RNN": COLOR_PHITD_LAPLACE,
    "PHI_TD_LOGISTIC_RNN": COLOR_PHITD_LOGISTIC,
    "PHI_TD_CAUCHY_RNN": COLOR_PHITD_CAUCHY,
    # Aliases / legacy names
    "MoG-PQN": COLOR_GREEN_MOG,
    "MoG-PQN-stab": COLOR_GREEN_MOG,
    # Sampling ablations (unique hues tied to family intent)
    "HALF_LAPLACIAN": COLOR_PHITD_LAPLACE,
    "EXPONENTIAL": COLOR_PHITD_LAPLACE,  # alias: truncated exponential ≡ half_laplacian
    "UNIFORM": COLOR_DARK_BLUE_PQN,
    "HALF_GAUSSIAN": COLOR_PURPLE_QTD,
    "PARETO_1": COLOR_PHITD_MOG,
    # φTD Phase-1 ablations: sequential hues within each one-factor sweep
    "N_OMEGA-32": "#94a3b8",
    "N_OMEGA-64": COLOR_MED_BLUE,
    "N_OMEGA-128": COLOR_PHITD_MOG,
    "N_OMEGA-256": "#7f1d1d",
    "M-10": "#94a3b8",
    "M-20": COLOR_MED_BLUE,
    "M-51": COLOR_PHITD_MOG,
    "M-100": "#7f1d1d",
    # Phase-2 combo tags — three family hues (red / purple / green), dark enough on white.
    # Within family: Pareto / Exponential / Uniform (paired with solid/dashed/dotted in plots).
    "ABL2-PhiTD-MoG-PARETO_1": "#b91c1c",
    "ABL2-PhiTD-MoG-EXPONENTIAL": "#ef4444",
    "ABL2-PhiTD-MoG-UNIFORM": "#7f1d1d",
    "ABL2-PhiTD-FCm-PARETO_1": "#7e22ce",
    "ABL2-PhiTD-FCm-EXPONENTIAL": "#a855f7",
    "ABL2-PhiTD-FCm-UNIFORM": "#581c87",
    "ABL2-PhiTD-FQm-PARETO_1": "#15803d",
    "ABL2-PhiTD-FQm-EXPONENTIAL": "#22c55e",
    "ABL2-PhiTD-FQm-UNIFORM": "#14532d",
}

# Lowercase lookup for robustness
_WANDB_TAG_HEX_LOWER = {k.lower(): v for k, v in _WANDB_TAG_HEX.items()}

_ALGO_KEY_TO_COLOR = {
    "mog": COLOR_GREEN_MOG,
    "iqn": COLOR_MED_BLUE,
    "ctd": COLOR_PINK_CTD,
    "qtd": COLOR_PURPLE_QTD,
    "pqn": COLOR_DARK_BLUE_PQN,
    "mog_cqn": COLOR_GREEN_MOG,
    "qr_dqn": COLOR_PURPLE_QTD,
    "c51": COLOR_PINK_CTD,
    "iqn_rnn": COLOR_MED_BLUE,
    "qtd_rnn": COLOR_PURPLE_QTD,
    "ctd_rnn": COLOR_PINK_CTD,
    "pqn_rnn": COLOR_DARK_BLUE_PQN,
    "mog_pqn_rnn": COLOR_GREEN_MOG,
    # φTD logical keys
    "phitd_fm": COLOR_PHITD_FM,
    "phitd_fcm": COLOR_PHITD_FCM,
    "phitd_fqm": COLOR_PHITD_FQM,
    "phitd_categorical": COLOR_PHITD_FCM,
    "phitd_quantile": COLOR_PHITD_FQM,
    "phitd_dirac": COLOR_PHITD_FM,
    "phitd_mog": COLOR_PHITD_MOG,
    "phitd_mogamma": COLOR_PHITD_MOGAMMA,
    "phitd_laplace": COLOR_PHITD_LAPLACE,
    "phitd_logistic": COLOR_PHITD_LOGISTIC,
    "phitd_cauchy": COLOR_PHITD_CAUCHY,
}

# Default legend strings (φ via mathtext). Mixture tags: TD-Gaussian, TD-Gamma (word), etc.
LEGEND_WANDB_TAG: dict[str, str] = {
    "PhiTD-Categorical": r"$\varphi\text{TD-Categorical}$",
    "PhiTD-Quantile": r"$\varphi\text{TD-Quantile}$",
    "PhiTD-Dirac": r"$\varphi\text{TD-Dirac}$",
    "PhiTD-Fm": r"$\varphi\text{TD-Dirac}$",
    "PhiTD-FCm": r"$\varphi\text{TD-Categorical}$",
    "PhiTD-FQm": r"$\varphi\text{TD-Quantile}$",
    "PhiTD-MoG": r"$\varphi\text{TD-Gaussian}$",
    "PhiTD-MoGamma": r"$\varphi\text{TD-Gamma}$",
    "PhiTD-Laplace": r"$\varphi\text{TD-Laplace}$",
    "PhiTD-Logistic": r"$\varphi\text{TD-Logistic}$",
    "PhiTD-Cauchy": r"$\varphi\text{TD-Cauchy}$",
    "PHI_TD_DIRAC_RNN": r"$\varphi\text{TD-Dirac}$",
    "PHI_TD_CATEGORICAL_RNN": r"$\varphi\text{TD-Categorical}$",
    "PHI_TD_QUANTILE_RNN": r"$\varphi\text{TD-Quantile}$",
    "PHI_TD_MOG_RNN": r"$\varphi\text{TD-Gaussian}$",
    "PHI_TD_GAMMA_RNN": r"$\varphi\text{TD-Gamma}$",
    "PHI_TD_LAPLACE_RNN": r"$\varphi\text{TD-Laplace}$",
    "PHI_TD_LOGISTIC_RNN": r"$\varphi\text{TD-Logistic}$",
    "PHI_TD_CAUCHY_RNN": r"$\varphi\text{TD-Cauchy}$",
    # Sampling scheme labels
    "PARETO_1": r"Pareto ($\alpha{=}1$)",
    "EXPONENTIAL": "Exponential",
    "HALF_LAPLACIAN": "Exponential",
    "UNIFORM": "Uniform",
    "HALF_GAUSSIAN": "Half-Gaussian",
    # Phase-1 one-factor sweeps
    "N_OMEGA-32": r"$N_\omega{=}32$",
    "N_OMEGA-64": r"$N_\omega{=}64$",
    "N_OMEGA-128": r"$N_\omega{=}128$",
    "N_OMEGA-256": r"$N_\omega{=}256$",
    "M-10": r"$m{=}10$",
    "M-20": r"$m{=}20$",
    "M-51": r"$m{=}51$",
    "M-100": r"$m{=}100$",
}

# ``plot_minatar_10m_phi_td_families`` only: same names plus $(F_{\cdot})$ class in parentheses.
LEGEND_PHI_TD_FAMILIES_COMPARE: dict[str, str] = {
    "PhiTD-Fm": r"$\varphi\text{TD-Dirac}$ $(F_m)$",
    "PhiTD-FCm": r"$\varphi\text{TD-Categorical}$ $(F_{C,m})$",
    "PhiTD-FQm": r"$\varphi\text{TD-Quantile}$ $(F_{Q,m})$",
}


def legend_label_for_wandb_algo_tag(tag: str, *, phi_families_compare: bool = False) -> str:
    """Pretty legend entry; φTD uses mathtext \\varphi."""
    if phi_families_compare:
        if tag in LEGEND_PHI_TD_FAMILIES_COMPARE:
            return LEGEND_PHI_TD_FAMILIES_COMPARE[tag]
        tl = tag.lower()
        for k, v in LEGEND_PHI_TD_FAMILIES_COMPARE.items():
            if k.lower() == tl:
                return v
    if tag in LEGEND_WANDB_TAG:
        return LEGEND_WANDB_TAG[tag]
    tl = tag.lower()
    for k, v in LEGEND_WANDB_TAG.items():
        if k.lower() == tl:
            return v
    # Phase-2 combo tags: ABL2-{family}-{sampling}
    if tag.startswith("ABL2-"):
        body = tag[len("ABL2-") :]
        for dist in (
            "PARETO_1",
            "EXPONENTIAL",
            "UNIFORM",
            "HALF_LAPLACIAN",
            "HALF_GAUSSIAN",
        ):
            suffix = f"-{dist}"
            if body.endswith(suffix):
                fam = body[: -len(suffix)]
                fam_lab = legend_label_for_wandb_algo_tag(fam)
                dist_lab = LEGEND_WANDB_TAG.get(dist, dist)
                return f"{fam_lab} / {dist_lab}"
    return tag


def algo_color(algo_key: str, default: str = "#444444") -> str:
    """Resolve plot color for a W&B algo tag or legacy palette key (case-insensitive for keys)."""
    s = algo_key.strip()
    if s in _WANDB_TAG_HEX:
        return _WANDB_TAG_HEX[s]
    sl = s.lower()
    if sl in _WANDB_TAG_HEX_LOWER:
        return _WANDB_TAG_HEX_LOWER[sl]
    # Phase-2 combo: color by family (sampling distinguished in legend text).
    if s.startswith("ABL2-"):
        body = s[len("ABL2-") :]
        for dist in (
            "PARETO_1",
            "EXPONENTIAL",
            "UNIFORM",
            "HALF_LAPLACIAN",
            "HALF_GAUSSIAN",
        ):
            suffix = f"-{dist}"
            if body.endswith(suffix):
                fam = body[: -len(suffix)]
                return algo_color(fam, default=default)
    return _ALGO_KEY_TO_COLOR.get(sl, default)


# Back-compat names (e.g. ``plot_sampling_distributions.py``)
COLOR_GREEN = COLOR_GREEN_MOG
COLOR_PURPLE = COLOR_PURPLE_QTD
COLOR_PINK = COLOR_PINK_CTD
COLOR_DARK_BLUE = COLOR_DARK_BLUE_PQN
