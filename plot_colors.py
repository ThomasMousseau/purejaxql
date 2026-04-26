"""Shared plotting palette for distributional algorithm comparisons."""

from __future__ import annotations

# Canonical colors requested for cross-plot consistency.
COLOR_DARK_BLUE = "#003a7d"  # MoG
COLOR_MED_BLUE = "#008dff"  # IQN
COLOR_PINK = "#ff73b6"  # CTD
COLOR_PURPLE = "#c701ff"  # QTD
COLOR_GREEN = "#4ecb8d"  # PQN / 5th algorithm when present

_ALGO_KEY_TO_COLOR = {
    # Canonical keys
    "mog": COLOR_DARK_BLUE,
    "iqn": COLOR_MED_BLUE,
    "ctd": COLOR_PINK,
    "qtd": COLOR_PURPLE,
    "pqn": COLOR_GREEN,
    # Common aliases
    "mog_cqn": COLOR_DARK_BLUE,
    "qr_dqn": COLOR_PURPLE,  # Legacy alias for QTD
    "c51": COLOR_PINK,  # Legacy alias for CTD
    "iqn_rnn": COLOR_MED_BLUE,
    "qtd_rnn": COLOR_PURPLE,
    "ctd_rnn": COLOR_PINK,
    "pqn_rnn": COLOR_GREEN,
    "mog_pqn_rnn": COLOR_DARK_BLUE,
}


def algo_color(algo_key: str, default: str = "#444444") -> str:
    """Resolve plot color for an algorithm key/tag/name."""
    return _ALGO_KEY_TO_COLOR.get(algo_key.lower(), default)

