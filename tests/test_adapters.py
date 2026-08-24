from __future__ import annotations

import numpy as np
import pytest

from alphaignitor.pipeline.zero_shot_ensemble.adapters import (
    ForecastResult,
    _extract_quantiles,
    _to_float_list,
    _to_optional_float_list,
)


class TestAdapters:
    def test_to_float_list(self):
        vals = [1.23, 4.56, "7.89"]
        res = _to_float_list(vals, horizon=3)
        assert res == [1.23, 4.56, 7.89]

        with pytest.raises(RuntimeError, match="Unexpected forecast length"):
            _to_float_list([1.0], horizon=3)

    def test_to_optional_float_list(self):
        vals = [1.0, None, float("nan"), "invalid", 5.0]
        res = _to_optional_float_list(vals, horizon=5)
        assert res == [1.0, None, None, None, 5.0]

    def test_extract_quantiles_3d(self):
        # Shape: (batch=2, horizon=3, quantiles=10)
        arr = np.zeros((2, 3, 10), dtype=np.float32)
        arr[0, :, 1] = 10.0  # q10
        arr[0, :, 5] = 50.0  # q50
        arr[0, :, 9] = 90.0  # q90

        q10, q50, q90 = _extract_quantiles(arr, batch_index=0, horizon=3)
        assert q10 == [10.0, 10.0, 10.0]
        assert q50 == [50.0, 50.0, 50.0]
        assert q90 == [90.0, 90.0, 90.0]

    def test_mock_adapter_flow(self, mock_adapters):
        adapter = mock_adapters["chronos2"]
        ctx = np.array([100.0, 101.0, 102.0], dtype=np.float32)
        res = adapter.forecast_result(ctx, horizon=3)

        assert isinstance(res, ForecastResult)
        assert len(res.point) == 3
        assert res.point[0] > 102.0  # multiplier applied
        assert len(res.q10) == 3
        assert len(res.q90) == 3

        batch_res = adapter.batch_forecast_result([ctx, ctx * 2], horizon=3)
        assert len(batch_res) == 2
