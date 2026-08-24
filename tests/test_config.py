from __future__ import annotations

from pathlib import Path

import pytest

from alphaignitor.config import PipelineConfig, load_pipeline_config


class TestConfig:
    def test_pipeline_config_defaults(self):
        cfg = PipelineConfig()
        assert cfg.prediction_days == 5
        assert cfg.context_days == 60
        assert cfg.optuna_window_days == 252
        assert cfg.optuna_n_trials == 100
        assert cfg.min_available_models == 2
        assert cfg.ensemble_models == ["chronos2", "timesfm", "tirex"]

    def test_load_pipeline_config_from_yaml(self, temp_dir: Path):
        yaml_file = temp_dir / "custom.yaml"
        yaml_file.write_text(
            """
            prediction_days: 10
            context_days: 45
            optuna_n_trials: 20
            ensemble_models:
              - chronos2
              - timesfm
            max_tickers: 50
            """,
            encoding="utf-8",
        )

        cfg = load_pipeline_config(yaml_file)
        assert cfg.prediction_days == 10
        assert cfg.context_days == 45
        assert cfg.optuna_n_trials == 20
        assert cfg.ensemble_models == ["chronos2", "timesfm"]
        assert cfg.max_tickers == 50

    def test_environment_variable_override_and_safety(self, temp_dir: Path, monkeypatch):
        yaml_file = temp_dir / "empty.yaml"
        yaml_file.write_text("{}", encoding="utf-8")

        monkeypatch.setenv("PREDICTION_DAYS", "7")
        monkeypatch.setenv("OPTUNA_N_TRIALS", "50")
        monkeypatch.setenv("MAX_TICKERS", "none")  # None handling
        monkeypatch.setenv("OPTIMIZER_WORKERS", "")  # Empty handling
        monkeypatch.setenv("ENSEMBLE_MODELS", "timesfm, tirex")

        cfg = load_pipeline_config(yaml_file)
        assert cfg.prediction_days == 7
        assert cfg.optuna_n_trials == 50
        assert cfg.max_tickers is None
        assert cfg.optimizer_workers is None
        assert cfg.ensemble_models == ["timesfm", "tirex"]
