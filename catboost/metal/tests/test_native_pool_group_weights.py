"""Read-only group weight access preserves raw and quantized Pool semantics."""
import os

import numpy as np
import pytest
from catboost import Pool

pytestmark = pytest.mark.skipif(os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
                               reason="requires the rebuilt native CatBoost Pool accessor")


def test_group_weights_stay_separate_from_object_weights():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    plain = Pool(x, np.arange(6))
    np.testing.assert_array_equal(plain.get_group_weight(), np.ones(6))
    group_weights = np.array([2, 2, 2, 0.5, 0.5, 0.5], dtype=np.float32)
    pool = Pool(x, np.arange(6), group_id=[0, 0, 0, 1, 1, 1], group_weight=group_weights)
    np.testing.assert_array_equal(pool.get_group_weight(), group_weights)
    np.testing.assert_array_equal(pool.get_weight(), np.ones(6))
    object_weights = np.linspace(1, 2, 6, dtype=np.float32)
    pool.set_weight(object_weights)
    np.testing.assert_array_equal(pool.get_weight(), object_weights)
    np.testing.assert_array_equal(pool.get_group_weight(), group_weights)


def test_group_weights_survive_quantized_storage_and_subset(tmp_path):
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    group_weights = np.array([2, 2, 2, 0.5, 0.5, 0.5], dtype=np.float32)
    pool = Pool(x, np.arange(6), group_id=[0, 0, 0, 1, 1, 1], group_weight=group_weights)
    pool.quantize(border_count=4)
    path = tmp_path / "groups.quantized"
    pool.save(path)
    restored = Pool("quantized://" + str(path))
    np.testing.assert_array_equal(restored.get_group_weight(), group_weights)
    np.testing.assert_array_equal(restored.slice([3, 4, 5]).get_group_weight(), group_weights[3:])
