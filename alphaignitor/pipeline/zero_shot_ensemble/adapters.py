from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class ForecastResult:
    point: list[float]
    q10: list[float | None]
    q50: list[float | None]
    q90: list[float | None]


class ZeroShotAdapter(ABC):
    name: str

    @abstractmethod
    def forecast(self, close_context: np.ndarray, *, horizon: int) -> list[float]:
        """Return close-price forecasts for horizons 1..horizon."""

    def forecast_result(self, close_context: np.ndarray, *, horizon: int) -> ForecastResult:
        point = self.forecast(close_context, horizon=horizon)
        none_quantiles = [None] * int(horizon)
        return ForecastResult(point=point, q10=none_quantiles[:], q50=none_quantiles[:], q90=none_quantiles[:])

    def batch_forecast(self, close_contexts: list[np.ndarray], *, horizon: int) -> list[list[float]]:
        """Return forecasts for multiple close-only contexts.

        Default implementation falls back to per-series inference.
        """
        return [self.forecast(ctx, horizon=horizon) for ctx in close_contexts]

    def batch_forecast_result(self, close_contexts: list[np.ndarray], *, horizon: int) -> list[ForecastResult]:
        return [self.forecast_result(ctx, horizon=horizon) for ctx in close_contexts]


class Chronos2Adapter(ZeroShotAdapter):
    name = "chronos2"

    def __init__(self, *, model_id: str = "amazon/chronos-2", device_map: str = "cuda") -> None:
        from chronos import Chronos2Pipeline

        self._pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device_map)

    def forecast(self, close_context: np.ndarray, *, horizon: int) -> list[float]:
        return self.forecast_result(close_context, horizon=horizon).point

    def forecast_result(self, close_context: np.ndarray, *, horizon: int) -> ForecastResult:
        context = np.asarray(close_context, dtype=np.float32)
        context_df = pd.DataFrame(
            {
                "id": "TICKER",
                "timestamp": pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(context), freq="B"),
                "target": context,
            }
        )
        pred_df = self._pipeline.predict_df(
            context_df,
            prediction_length=int(horizon),
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        ordered = pred_df.sort_values("timestamp").reset_index(drop=True)
        point_col = "predictions" if "predictions" in ordered.columns else "0.5"
        point = _to_float_list(ordered[point_col], horizon=horizon)
        q10 = _to_optional_float_list(ordered["0.1"], horizon=horizon) if "0.1" in ordered.columns else [None] * int(horizon)
        q50 = _to_optional_float_list(ordered["0.5"], horizon=horizon) if "0.5" in ordered.columns else [None] * int(horizon)
        q90 = _to_optional_float_list(ordered["0.9"], horizon=horizon) if "0.9" in ordered.columns else [None] * int(horizon)
        return ForecastResult(point=point, q10=q10, q50=q50, q90=q90)

    def batch_forecast(self, close_contexts: list[np.ndarray], *, horizon: int) -> list[list[float]]:
        return [result.point for result in self.batch_forecast_result(close_contexts, horizon=horizon)]

    def batch_forecast_result(self, close_contexts: list[np.ndarray], *, horizon: int) -> list[ForecastResult]:
        if not close_contexts:
            return []
        rows: list[pd.DataFrame] = []
        for idx, context in enumerate(close_contexts):
            values = np.asarray(context, dtype=np.float32)
            rows.append(
                pd.DataFrame(
                    {
                        "id": f"T{idx}",
                        "timestamp": pd.date_range(
                            end=pd.Timestamp.today().normalize(),
                            periods=len(values),
                            freq="B",
                        ),
                        "target": values,
                    }
                )
            )
        context_df = pd.concat(rows, ignore_index=True)
        pred_df = self._pipeline.predict_df(
            context_df,
            prediction_length=int(horizon),
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        if "id" not in pred_df.columns:
            return [self.forecast_result(ctx, horizon=horizon) for ctx in close_contexts]
        ordered = pred_df.sort_values(["id", "timestamp"]).reset_index(drop=True)
        out: list[list[float]] = []
        point_col = "predictions" if "predictions" in ordered.columns else "0.5"
        out_results: list[ForecastResult] = []
        for idx in range(len(close_contexts)):
            group = ordered[ordered["id"] == f"T{idx}"]
            point = _to_float_list(group[point_col], horizon=horizon)
            q10 = _to_optional_float_list(group["0.1"], horizon=horizon) if "0.1" in group.columns else [None] * int(horizon)
            q50 = _to_optional_float_list(group["0.5"], horizon=horizon) if "0.5" in group.columns else [None] * int(horizon)
            q90 = _to_optional_float_list(group["0.9"], horizon=horizon) if "0.9" in group.columns else [None] * int(horizon)
            out_results.append(ForecastResult(point=point, q10=q10, q50=q50, q90=q90))
        return out_results


class TimesFMAdapter(ZeroShotAdapter):
    name = "timesfm"

    def __init__(
        self,
        *,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        context_days: int = 60,
        horizon: int = 3,
        device: str = "auto",
    ) -> None:
        import timesfm

        torch.set_float32_matmul_precision("high")
        self._timesfm = timesfm
        try:
            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id, device=device)
        except TypeError:
            # Backward compatibility for TimesFM versions without explicit `device` argument.
            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=int(context_days),
                max_horizon=int(horizon),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    def forecast(self, close_context: np.ndarray, *, horizon: int) -> list[float]:
        return self.forecast_result(close_context, horizon=horizon).point

    def forecast_result(self, close_context: np.ndarray, *, horizon: int) -> ForecastResult:
        context = np.asarray(close_context, dtype=np.float32)
        point_forecast, quantiles = self._model.forecast(horizon=int(horizon), inputs=[context])
        point = _to_float_list(np.asarray(point_forecast, dtype=np.float32).reshape(-1), horizon=horizon)
        q10, q50, q90 = _extract_quantiles(quantiles, batch_index=0, horizon=horizon)
        return ForecastResult(point=point, q10=q10, q50=q50, q90=q90)

    def batch_forecast(self, close_contexts: list[np.ndarray], *, horizon: int) -> list[list[float]]:
        return [result.point for result in self.batch_forecast_result(close_contexts, horizon=horizon)]

    def batch_forecast_result(self, close_contexts: list[np.ndarray], *, horizon: int) -> list[ForecastResult]:
        if not close_contexts:
            return []
        contexts = [np.asarray(ctx, dtype=np.float32) for ctx in close_contexts]
        point_forecast, quantiles = self._model.forecast(horizon=int(horizon), inputs=contexts)
        arr = np.asarray(point_forecast, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out: list[ForecastResult] = []
        for idx in range(arr.shape[0]):
            point = _to_float_list(arr[idx], horizon=horizon)
            q10, q50, q90 = _extract_quantiles(quantiles, batch_index=idx, horizon=horizon)
            out.append(ForecastResult(point=point, q10=q10, q50=q50, q90=q90))
        return out


class TirexAdapter(ZeroShotAdapter):
    name = "tirex"

    def __init__(self, *, model_id: str = "NX-AI/TiRex", backend: str = "auto") -> None:
        from tirex import load_model

        kwargs = {} if backend == "auto" else {"backend": backend}
        self._model = load_model(model_id, **kwargs)

    def forecast(self, close_context: np.ndarray, *, horizon: int) -> list[float]:
        return self.forecast_result(close_context, horizon=horizon).point

    def forecast_result(self, close_context: np.ndarray, *, horizon: int) -> ForecastResult:
        context = torch.from_numpy(np.ascontiguousarray(close_context, dtype=np.float32)).unsqueeze(0)
        quantiles, mean = self._model.forecast(
            context=context,
            prediction_length=int(horizon),
            output_type="numpy",
        )
        point = _to_float_list(np.asarray(mean, dtype=np.float32).reshape(-1), horizon=horizon)
        q10, q50, q90 = _extract_quantiles(quantiles, batch_index=0, horizon=horizon)
        return ForecastResult(point=point, q10=q10, q50=q50, q90=q90)

    def batch_forecast(self, close_contexts: list[np.ndarray], *, horizon: int) -> list[list[float]]:
        return [result.point for result in self.batch_forecast_result(close_contexts, horizon=horizon)]

    def batch_forecast_result(self, close_contexts: list[np.ndarray], *, horizon: int) -> list[ForecastResult]:
        if not close_contexts:
            return []
        matrix = np.stack([np.ascontiguousarray(ctx, dtype=np.float32) for ctx in close_contexts], axis=0)
        context = torch.from_numpy(matrix)
        quantiles, mean = self._model.forecast(
            context=context,
            prediction_length=int(horizon),
            output_type="numpy",
        )
        arr = np.asarray(mean, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out: list[ForecastResult] = []
        for idx in range(arr.shape[0]):
            point = _to_float_list(arr[idx], horizon=horizon)
            q10, q50, q90 = _extract_quantiles(quantiles, batch_index=idx, horizon=horizon)
            out.append(ForecastResult(point=point, q10=q10, q50=q50, q90=q90))
ADAPTER_REGISTRY: dict[str, type[ZeroShotAdapter]] = {
    "chronos2": Chronos2Adapter,
    "timesfm": TimesFMAdapter,
    "tirex": TirexAdapter,
}


def register_adapter(name: str, adapter_cls: type[ZeroShotAdapter]) -> None:
    """Register a custom ZeroShotAdapter class."""
    ADAPTER_REGISTRY[name.strip().lower()] = adapter_cls


def load_adapters(
    models: list[str],
    *,
    context_days: int,
    horizon: int,
    chronos2_device_map: str = "cuda",
    timesfm_device: str = "auto",
    tirex_backend: str = "auto",
    allow_cpu: bool = False,
) -> dict[str, ZeroShotAdapter]:
    adapters: dict[str, ZeroShotAdapter] = {}
    is_cpu_allowed = allow_cpu or os.environ.get("ALLOW_CPU_FOR_TESTING") == "1"
    if not is_cpu_allowed:
        _require_gpu_runtime(
            chronos2_device_map=chronos2_device_map,
            timesfm_device=timesfm_device,
            tirex_backend=tirex_backend,
        )
    _configure_safe_sdpa_backends()
    resolved_chronos2_device = _resolve_chronos2_device_map(chronos2_device_map)
    for model in models:
        model_key = model.strip().lower()
        if model_key == "chronos2":
            adapters[model] = Chronos2Adapter(device_map=resolved_chronos2_device)
        elif model_key == "timesfm":
            adapters[model] = TimesFMAdapter(
                context_days=context_days,
                horizon=horizon,
                device=timesfm_device,
            )
        elif model_key == "tirex":
            adapters[model] = TirexAdapter(backend=tirex_backend)
        elif model_key in ADAPTER_REGISTRY:
            adapters[model] = ADAPTER_REGISTRY[model_key]()
        else:
            raise ValueError(f"Unsupported zero-shot model: {model}")
    return adapters


def _require_gpu_runtime(
    *,
    chronos2_device_map: str,
    timesfm_device: str,
    tirex_backend: str,
) -> None:
    # CPU execution is explicitly disallowed for production pipeline.
    if str(chronos2_device_map).strip().lower() == "cpu":
        raise RuntimeError("chronos2_device_map=cpu is not allowed. GPU execution is required.")
    if str(timesfm_device).strip().lower() == "cpu":
        raise RuntimeError("timesfm_device=cpu is not allowed. GPU execution is required.")
    if str(tirex_backend).strip().lower() == "cpu":
        raise RuntimeError("tirex_backend=cpu is not allowed. GPU execution is required.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU runtime is unavailable. CPU fallback is disabled by policy. "
            "Ensure ROCm/CUDA GPU is available before running forecast."
        )

    try:
        _ = torch.tensor([0.0], device="cuda")
    except Exception as e:
        raise RuntimeError(
            "GPU runtime check failed. CPU fallback is disabled by policy. "
            "Ensure ROCm/CUDA GPU is usable before running forecast."
        ) from e


def _configure_safe_sdpa_backends() -> None:
    # Keep ROCm attention on stable kernels only.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _resolve_chronos2_device_map(device_map: str) -> str:
    value = str(device_map).strip().lower()
    if value in {"", "auto"}:
        return "cuda"
    if value.startswith("cuda"):
        return value
    return str(device_map)


def _to_float_list(values, *, horizon: int) -> list[float]:
    out = [float(v) for v in list(values)[: int(horizon)]]
    if len(out) != int(horizon):
        raise RuntimeError(f"Unexpected forecast length: got {len(out)}, expected {horizon}")
    return out


def _to_optional_float_list(values, *, horizon: int) -> list[float | None]:
    raw = list(values)[: int(horizon)]
    if len(raw) != int(horizon):
        raise RuntimeError(f"Unexpected forecast length: got {len(raw)}, expected {horizon}")
    out: list[float | None] = []
    for value in raw:
        try:
            v = float(value)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(v if np.isfinite(v) else None)
    return out


def _extract_quantiles(quantiles, *, batch_index: int, horizon: int) -> tuple[list[float | None], list[float | None], list[float | None]]:
    empty = ([None] * int(horizon), [None] * int(horizon), [None] * int(horizon))
    if quantiles is None:
        return empty

    arr = np.asarray(quantiles, dtype=np.float32)
    if arr.size == 0:
        return empty

    # Extract 2D slice for the specific batch if 3D
    if arr.ndim == 3:
        if batch_index >= arr.shape[0]:
            return empty
        sub = arr[batch_index]
    elif arr.ndim == 2:
        sub = arr
    else:
        return empty

    dim1, dim2 = sub.shape

    # Case A: (horizon, num_quantiles)
    if dim1 == int(horizon):
        if dim2 >= 10:
            i10, i50, i90 = 1, 5, 9
        elif dim2 >= 9:
            i10, i50, i90 = 0, 4, 8
        elif dim2 >= 3:
            i10, i50, i90 = 0, 1, 2
        else:
            return empty
        return (
            _to_optional_float_list(sub[:, i10], horizon=horizon),
            _to_optional_float_list(sub[:, i50], horizon=horizon),
            _to_optional_float_list(sub[:, i90], horizon=horizon),
        )

    # Case B: (num_quantiles, horizon)
    if dim2 == int(horizon):
        if dim1 >= 10:
            i10, i50, i90 = 1, 5, 9
        elif dim1 >= 9:
            i10, i50, i90 = 0, 4, 8
        elif dim1 >= 3:
            i10, i50, i90 = 0, 1, 2
        else:
            # Fallback if we only have 1 quantile (like median/mean)
            return empty[0], _to_optional_float_list(sub[0], horizon=horizon), empty[2]
        return (
            _to_optional_float_list(sub[i10, :], horizon=horizon),
            _to_optional_float_list(sub[i50, :], horizon=horizon),
            _to_optional_float_list(sub[i90, :], horizon=horizon),
        )

    return empty
