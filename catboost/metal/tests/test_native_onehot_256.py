"""Native one-hot bin 255 is a known value, separate from unseen inference."""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, Pool
from catboost_metal._categorical import cat_feature_hashes

from test_native_feature_weights import options


pytestmark = pytest.mark.skipif(os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
                               reason="requires the native Metal 256-category boundary")


@pytest.mark.parametrize("mode", ["PlainDP", "PlainFP", "OrderedFP", "Depthwise", "Lossguide", "Region"])
@pytest.mark.parametrize("quantized", [False, True])
def test_known_bin_255_is_selected_and_unseen_value_takes_other_branch(mode, quantized, tmp_path):
    categories = np.array([f"category-{i}" for i in range(256)], object)
    hashes = cat_feature_hashes(categories)
    assert len(np.unique(hashes)) == 256
    # Native categorical.cpp sorts original uint32 hashes before assigning
    # dense bins, so this category specifically exercises uint8 value 255.
    special = int(np.argmax(hashes))
    x = np.tile(categories, 4).reshape(-1, 1)
    y = np.tile((np.arange(256) == special).astype(np.float32), 4)
    pool = Pool(x, y, cat_features=[0])
    if quantized:
        pool.quantize(border_count=1)
    model = CatBoost(options(mode, one_hot_max_size=256, iterations=1, depth=1)).fit(pool)
    assert model.get_metadata()["metal_backend"] == "METAL"
    file = tmp_path / "onehot256.json"
    model.save_model(file, format="json")
    document = json.loads(file.read_text())
    split = (document["oblivious_trees"][0]["splits"][0] if "oblivious_trees" in document
             else document["trees"][0]["split"])
    assert split["split_type"] == "OneHotFeature"
    assert split["value"] & 0xffffffff == int(hashes[special])
    future = [[categories[special]], [categories[(special + 1) % 256]], ["previously-unseen-category"]]
    expected = model.predict(future)
    assert expected[0] > expected[1]
    assert expected[1] == expected[2]
    np.testing.assert_allclose(model.predict(future, task_type="GPU"), expected, rtol=2e-6, atol=2e-6)
    restored = CatBoost().load_model(file, format="json")
    np.testing.assert_array_equal(restored.predict(future), expected)
    np.testing.assert_allclose(restored.predict(future, task_type="GPU"), expected, rtol=2e-6, atol=2e-6)
