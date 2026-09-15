"""CUDA bundle-manager IDs on CPU-prequantized Pools; every fit uses Metal."""
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool

from test_native_feature_weights import options, root_feature


pytestmark = pytest.mark.skipif(os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal bundle source metadata")


@pytest.fixture(autouse=True)
def only_gpu_fits(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def bundled_problem(*, classes=False, grouped=False):
    row = np.arange(840)
    category, control = row % 21, (row // 21) % 2
    # Twenty mutually exclusive binary columns form one retained EFB. Smaller
    # binary-only bundles can be discarded by the CPU quantizer in favor of a
    # binary pack (exclusive_feature_bundling.cpp). The overlapping control
    # cannot join this bundle with the default zero allowed conflicts.
    x = np.column_stack([category == feature for feature in range(20)] + [control]).astype(np.float32)
    y = (2 * x[:, 0] + control).astype(np.int32) if classes else 8 * x[:, 0] + .5 * control
    pool = Pool(x, y, **({"group_id": row // 42} if grouped else {}))
    pool.quantize(task_type="CPU", border_count=1, sparse_features_conflict_fraction=0)
    return pool


@pytest.mark.parametrize("path", ["PlainDP", "PlainFP", "OrderedFP", "Depthwise", "Lossguide", "Region"])
def test_bundle_weight_addresses_shared_manager_and_original_weights_are_unused(path, tmp_path):
    pool = bundled_problem()
    ordinary = CatBoost(options(path)).fit(pool)
    original_weight = CatBoost(options(path, feature_weights={0: 0})).fit(pool)
    # The 21 originals retain their IDs. The sole bundle registers next as 21,
    # and all twenty component predicates receive that shared source weight.
    bundle_weight = CatBoost(options(path, feature_weights={21: 0})).fit(pool)
    assert root_feature(ordinary, tmp_path, "ordinary") == 0
    assert root_feature(original_weight, tmp_path, "original") == 0
    assert root_feature(bundle_weight, tmp_path, "bundle") == 20
    np.testing.assert_array_equal(original_weight.get_leaf_values(), ordinary.get_leaf_values())
    np.testing.assert_array_equal(original_weight.predict(pool, task_type="GPU"),
                                  ordinary.predict(pool, task_type="GPU"))


@pytest.mark.parametrize("loss", ["MultiClass", "PairLogitPairwise"])
def test_doc_only_losses_omit_bundle_registry_ids_from_quantized_pools(loss):
    pool = bundled_problem(classes=loss == "MultiClass", grouped=loss == "PairLogitPairwise")
    # CUDA deliberately drops CPU bundle metadata for these losses, so no
    # manager 21 exists. The same key above must fail the registry bounds check.
    config = options(loss_function=loss, feature_weights={21: 0},
                     score_function="L2" if loss == "MultiClass" else "NewtonL2", leaf_estimation_method="Newton")
    with pytest.raises(CatBoostError, match="feature|Feature"):
        CatBoost(config).fit(pool)
