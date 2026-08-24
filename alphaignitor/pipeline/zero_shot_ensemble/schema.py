from __future__ import annotations

import json
import math

ENSEMBLE_MODELS = ["chronos2", "timesfm", "tirex"]
FORECAST_HORIZONS = [1, 2, 3, 4, 5]

METRIC_TARGETS = ENSEMBLE_MODELS + ["ensemble"]
METRIC_NAMES = ["directional_accuracy", "rmse", "mae"]

CORE_FORECAST_COLUMNS = [
    "ticker",
    "asof_trade_date",
    "horizon",
    "forecast_trade_date",
    "asof_close",
    "chronos2_pred",
    "timesfm_pred",
    "tirex_pred",
    "ensemble_pred",
    "ensemble_return",
    "ensemble_direction",
    "actual_close",
    "actual_direction",
    "weight_chronos2",
    "weight_timesfm",
    "weight_tirex",
    "weights_json",
    "chronos2_directional_accuracy",
    "chronos2_rmse",
    "chronos2_mae",
    "timesfm_directional_accuracy",
    "timesfm_rmse",
    "timesfm_mae",
    "tirex_directional_accuracy",
    "tirex_rmse",
    "tirex_mae",
    "ensemble_directional_accuracy",
    "ensemble_rmse",
    "ensemble_mae",
    # Legacy aliases for old chart/report helpers. They are log returns from as-of close.
    "q0.1",
    "q0.5",
    "q0.9",
]


def normalize_weights(weights: dict[str, float], *, available_models: set[str] | None = None) -> dict[str, float]:
    """Return non-negative weights normalized to sum to 1.

    `available_models` is used when one zero-shot model fails. In that case the
    remaining model weights are re-normalized instead of failing the whole ticker.
    """
    allowed = available_models if available_models is not None else set(weights)
    clean: dict[str, float] = {}
    for model, value in weights.items():
        if model not in allowed:
            continue
        v = float(value)
        if math.isfinite(v) and v > 0:
            clean[model] = v

    total = sum(clean.values())
    if total <= 0:
        models = sorted(allowed)
        if not models:
            return {}
        equal = 1.0 / len(models)
        return {model: equal for model in models}
    return {model: value / total for model, value in clean.items()}


def equal_weights(models: list[str]) -> dict[str, float]:
    if not models:
        return {}
    w = 1.0 / len(models)
    return {model: w for model in models}


def weights_to_json(weights: dict[str, float]) -> str:
    return json.dumps({k: round(float(v), 8) for k, v in sorted(weights.items())}, ensure_ascii=False)


def direction_from_delta(delta: float, *, eps: float = 1e-12) -> int | None:
    if not math.isfinite(delta) or abs(delta) <= eps:
        return None
    return 1 if delta > 0 else -1


def evaluable_direction_pair(asof_close: float, pred_close: float, actual_close: float) -> tuple[int, int] | None:
    """Return predicted/actual directions, excluding zero and flat evaluation cases."""
    values = [float(asof_close), float(pred_close), float(actual_close)]
    if not all(math.isfinite(v) for v in values):
        return None
    if float(pred_close) == 0.0 or float(actual_close) == 0.0:
        return None
    pred_dir = direction_from_delta(float(pred_close) - float(asof_close))
    actual_dir = direction_from_delta(float(actual_close) - float(asof_close))
    if pred_dir is None or actual_dir is None:
        return None
    return pred_dir, actual_dir


def display_direction(asof_close: float, close_value: float | None) -> int | None:
    """Return UP/DOWN direction for display, excluding zero close values."""
    if close_value is None:
        return None
    try:
        asof = float(asof_close)
        value = float(close_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(asof) or not math.isfinite(value) or value == 0.0:
        return None
    return direction_from_delta(value - asof)


def log_return(pred_close: float, asof_close: float) -> float:
    if pred_close > 0 and asof_close > 0:
        return math.log(pred_close / asof_close)
    return float("nan")
