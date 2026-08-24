"""パイプライン設定。

YAML ファイルと環境変数の両方から設定を読み込む。
環境変数は YAML の値を上書きする。
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class PipelineConfig:
    run_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    prediction_days: int = 5
    context_days: int = 60
    optuna_window_days: int = 252
    optuna_n_trials: int = 100
    optuna_timeout_minutes: int = 10
    optuna_storage_path: str = "models/ensemble_optuna.sqlite3"
    prediction_cache_path: str = "cache/zero_shot_predictions.sqlite3"
    ensemble_models: list[str] = dataclasses.field(default_factory=lambda: ["chronos2", "timesfm", "tirex"])
    chronos2_device_map: str = "auto"
    timesfm_device: str = "auto"
    tirex_backend: str = "auto"
    chronos2_batch_size: int = 64
    timesfm_batch_size: int = 256
    tirex_batch_size: int = 256
    min_available_models: int = 2
    optimizer_workers: int | None = None
    top_n: int = 10
    max_tickers: int | None = None
    model_root: str = "models"
    report_outdir: str = "report"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "prod.yaml"


def _parse_env_value(field_type: object, env_val: str) -> object:
    val = env_val.strip()
    if val.lower() in {"none", "null", ""}:
        return None
    type_str = str(field_type).lower()
    if "bool" in type_str:
        return val.lower() in {"1", "true", "yes", "on"}
    if "int" in type_str:
        try:
            return int(val)
        except ValueError:
            return None
    if "float" in type_str:
        try:
            return float(val)
        except ValueError:
            return None
    if "list" in type_str:
        return [s.strip() for s in val.split(",") if s.strip()]
    return val


def load_pipeline_config(path: Path) -> PipelineConfig:
    """YAML ファイルを読み込み、環境変数で上書きして PipelineConfig を返す。"""
    data: dict = {}
    if path.exists():
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config. Install pyyaml.")
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            data = dict(loaded)

    valid_fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    cfg = PipelineConfig(**{k: v for k, v in data.items() if k in valid_fields})

    # 環境変数で設定値を上書きする
    for field in dataclasses.fields(PipelineConfig):
        env_val = os.environ.get(field.name.upper())
        if env_val is not None:
            parsed = _parse_env_value(field.type, env_val)
            setattr(cfg, field.name, parsed)

    return cfg
