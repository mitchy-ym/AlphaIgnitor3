from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from alphaignitor.pipeline.zero_shot_ensemble.metrics import (
    directional_accuracy,
    mae,
    metric_summary,
    rmse,
)
from alphaignitor.pipeline.zero_shot_ensemble.schema import (
    direction_from_delta,
    display_direction,
    equal_weights,
    evaluable_direction_pair,
    log_return,
    normalize_weights,
)


class TestSchemaAndMetrics:
    def test_direction_from_delta(self):
        assert direction_from_delta(1.5) == 1
        assert direction_from_delta(-0.8) == -1
        assert direction_from_delta(0.0) is None
        assert direction_from_delta(float("nan")) is None

    def test_evaluable_direction_pair(self):
        # asof=100, pred=105 (UP), actual=102 (UP) -> (1, 1)
        assert evaluable_direction_pair(100.0, 105.0, 102.0) == (1, 1)
        # asof=100, pred=95 (DOWN), actual=102 (UP) -> (-1, 1)
        assert evaluable_direction_pair(100.0, 95.0, 102.0) == (-1, 1)
        # Flat -> None
        assert evaluable_direction_pair(100.0, 100.0, 102.0) is None

    def test_display_direction(self):
        assert display_direction(100.0, 105.0) == 1
        assert display_direction(100.0, 95.0) == -1
        assert display_direction(100.0, None) is None
        assert display_direction(100.0, 0.0) is None

    def test_log_return(self):
        ret = log_return(110.0, 100.0)
        assert math.isclose(ret, math.log(1.1), rel_tol=1e-6)
        assert math.isnan(log_return(-1.0, 100.0))

    def test_normalize_weights(self):
        raw = {"chronos2": 0.2, "timesfm": 0.3, "tirex": 0.5}
        norm = normalize_weights(raw)
        assert math.isclose(sum(norm.values()), 1.0)
        assert math.isclose(norm["chronos2"], 0.2)
        assert math.isclose(norm["timesfm"], 0.3)
        assert math.isclose(norm["tirex"], 0.5)

        # Subset with available_models
        partial = normalize_weights(raw, available_models={"chronos2", "timesfm"})
        assert set(partial.keys()) == {"chronos2", "timesfm"}
        assert math.isclose(sum(partial.values()), 1.0)
        assert math.isclose(partial["chronos2"], 0.4)
        assert math.isclose(partial["timesfm"], 0.6)

        # All zeros -> fallback equal weights
        zeros = normalize_weights({"a": 0.0, "b": 0.0})
        assert zeros == {"a": 0.5, "b": 0.5}

    def test_equal_weights(self):
        eq = equal_weights(["a", "b", "c"])
        assert eq == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        assert equal_weights([]) == {}

    def test_directional_accuracy(self):
        # 3 correct, 1 wrong
        rows = [
            (100.0, 105.0, 103.0),  # hit UP/UP
            (100.0, 95.0, 92.0),    # hit DOWN/DOWN
            (100.0, 102.0, 101.0),  # hit UP/UP
            (100.0, 105.0, 95.0),   # miss UP/DOWN
        ]
        assert math.isclose(directional_accuracy(rows), 0.75)

    def test_rmse_and_mae(self):
        pred = pd.Series([100.0, 102.0, 104.0])
        actual = pd.Series([98.0, 102.0, 100.0])
        # diffs: 2, 0, 4 -> mean square = (4 + 0 + 16)/3 = 20/3
        expected_rmse = math.sqrt(20 / 3)
        expected_mae = (2 + 0 + 4) / 3

        assert math.isclose(rmse(pred, actual), expected_rmse, rel_tol=1e-5)
        assert math.isclose(mae(pred, actual), expected_mae, rel_tol=1e-5)

    def test_metric_summary(self):
        df = pd.DataFrame(
            {
                "asof_close": [100.0, 100.0],
                "pred": [105.0, 95.0],
                "actual_close": [102.0, 90.0],
            }
        )
        summary = metric_summary(df, pred_col="pred")
        assert summary["directional_accuracy"] == 1.0
        assert not math.isnan(summary["rmse"])
        assert not math.isnan(summary["mae"])
