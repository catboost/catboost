"""CUDA packed-feature RSM and shared host draws, without CPU model fitting."""
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool

from test_native_compound_ctrs import exported, snapshot_options
from test_native_greedy_api import StopAfter
from test_ordered_rng import ReferenceMt64


pytestmark = pytest.mark.skipif(os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal rsm adapter")
LOSSES = ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise")


@pytest.fixture(autouse=True)
def only_gpu_fits(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def reference_masks(folds, *, loss, seed=761, rsm=.23, histories=1, simple=False,
                    bootstrap="No", subsample=.7, trees=5, ctr=True):
    """Independent MT interpreter and literal single-device CUDA draw schedule."""
    grids = [[i for i, count in enumerate(folds) if count == 1],
             [i for i, count in enumerate(folds) if 1 < count <= 15],
             [i for i, count in enumerate(folds) if count > 15]]
    ordering = ReferenceMt64(0)
    for index in range(1, len(grids[2])):
        other = ordering.uniform(index + 1)
        grids[2][index], grids[2][other] = grids[2][other], grids[2][index]
    # This fixture has <=16 OneByte entries; CUDA libc++ insertion sort keeps
    # equal grouping levels in their shuffled order for this small grid.
    grids[2].sort(key=lambda feature: (folds[feature] + 1) / 256 + (folds[feature] + 1 > 129))
    if ctr and histories == 1:
        grids[1].append(len(folds))
    grids += [[], [len(folds)] if ctr and histories > 1 else [], []]
    random = ReferenceMt64(seed)
    random.advance(1)
    draws, warm, result = 1, False, []
    for _ in range(trees):
        if not warm and (loss != "QueryCrossEntropy" or bootstrap == "Bernoulli" and subsample < 1):
            random.advance(65537)
            draws += 65537
            warm = True
        if loss == "YetiRankPairwise":
            random.advance(1)
            draws += 1
        active = set(grids[0])
        for policy, per_pack in ((1, 8), (2, 4), (4, 8), (5, 4)):
            grid = grids[policy]
            if not grid:
                continue
            probability = rsm
            if rsm == 1:
                active.update(grid)
                continue
            while True:
                selected = []
                for first in range(0, len(grid), per_pack):
                    uniform = (random.next() >> 11) * (1.0 / ((1 << 53) - 1))
                    draws += 1
                    if uniform <= probability:
                        selected.extend(grid[first:first + per_pack])
                if selected:
                    active.update(selected)
                    break
                probability = min(2 * probability, 1)
        result.append(active)
        if loss == "YetiRankPairwise" and not simple:
            random.advance(histories)
            draws += histories
    return result, draws


def problem(tmp_path, *, binary_only=False):
    rng = np.random.default_rng(83271)
    requested = [1] * 17 if binary_only else [1] + [3] * 16 + [31] * 16
    rows = 128
    x = np.column_stack([rng.permutation(np.arange(rows) % (count + 1)) for count in requested]).astype(float)
    labels = (np.arange(rows) % 4) / 3
    groups = np.repeat(np.arange(rows // 8), 8)
    pairs = [(first + 3, first) for first in range(0, rows, 4)]
    if binary_only:
        pool = Pool(x, labels, group_id=groups, pairs=pairs)
    else:
        mixed = x.astype(object)
        mixed = np.column_stack([mixed, np.array([f"c{row % 7}" for row in range(rows)], object)])
        pool = Pool(mixed, labels, group_id=groups, pairs=pairs, cat_features=[len(requested)])
    pool.quantize(per_float_feature_quantization=[f"{i}:border_count={count}" for i, count in enumerate(requested)])
    path = tmp_path / "rsm-borders.tsv"
    pool.save_quantization_borders(str(path))
    folds = np.zeros(len(requested), int)
    for line in path.read_text().splitlines():
        folds[int(line.split("\t")[0])] += 1
    np.testing.assert_array_equal(folds, requested)
    return pool, folds


def options(loss, histories=1, method="Newton", sampling="No", **extra):
    result = dict(task_type="GPU", loss_function=loss, boosting_type="Plain", data_partition="DocParallel",
        grow_policy="SymmetricTree", iterations=5, depth=2, learning_rate=.15, l2_leaf_reg=2,
        random_strength=0, random_seed=761, rsm=.23, bootstrap_type=sampling,
        leaf_estimation_method=method, leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
        score_function="NewtonL2", permutation_count=histories, has_time=histories == 1,
        verbose=False, allow_writing_files=False, max_ctr_complexity=1, one_hot_max_size=2,
        simple_ctr=["Borders:CtrBorderCount=3:Prior=0.5"], ctr_target_border_count=1)
    if sampling == "Bernoulli": result["subsample"] = .7
    return result | extra


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("histories", [1, 4])
@pytest.mark.parametrize("method", ["Newton", "Simple"])
@pytest.mark.parametrize("sampling", ["No", "Bernoulli"])
def test_every_depth_uses_the_source_packed_grid_and_exact_host_draws(tmp_path, loss, histories, method, sampling):
    pool, folds = problem(tmp_path)
    model = CatBoost(options(loss, histories, method, sampling)).fit(pool)
    expected, draws = reference_masks(folds, loss=loss, histories=histories,
        simple=method == "Simple", bootstrap=sampling)
    assert len({tuple(sorted(mask)) for mask in expected}) > 1
    assert all(0 in mask for mask in expected)
    assert any(len(mask) < len(folds) for mask in expected)
    document = exported(model, tmp_path / "rsm.json")
    for tree, mask in zip(document["oblivious_trees"], expected):
        for split in tree["splits"]:
            assert (split["float_feature_index"] if split["split_type"] == "FloatFeature" else len(folds)) in mask
    assert model.get_metadata()["metal_rsm_rng"] == "cuda_single_device_host_shadow_v1"
    assert int(model.get_metadata()["metal_rsm_host_draw_count"]) == draws


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("histories", [1, 4])
def test_snapshot_host_replay_preserves_sampled_forest_and_draw_count(tmp_path, loss, histories):
    pool, _ = problem(tmp_path)
    config = options(loss, histories)
    complete = CatBoost(config).fit(pool, eval_set=pool, use_best_model=False)
    saved = snapshot_options(config, tmp_path / "resume")
    partial = CatBoost(saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(2)])
    assert partial.tree_count_ == 2
    resumed = CatBoost(saved).fit(pool, eval_set=pool, use_best_model=False)
    for method in ("get_leaf_values", "get_leaf_weights", "get_tree_leaf_counts"):
        np.testing.assert_array_equal(getattr(resumed, method)(), getattr(complete, method)())
    np.testing.assert_array_equal(resumed.predict(pool, task_type="GPU"), complete.predict(pool, task_type="GPU"))
    assert resumed.get_evals_result() == complete.get_evals_result()
    assert resumed.get_metadata()["metal_rsm_host_draw_count"] == complete.get_metadata()["metal_rsm_host_draw_count"]


@pytest.mark.parametrize("loss", LOSSES)
def test_binary_features_are_always_retained_even_below_nonbinary_rsm_threshold(tmp_path, loss):
    pool, _ = problem(tmp_path, binary_only=True)
    sampled = CatBoost(options(loss, rsm=.005)).fit(pool)
    ordinary = CatBoost(options(loss, rsm=1)).fit(pool)
    np.testing.assert_array_equal(sampled.get_leaf_values(), ordinary.get_leaf_values())
    np.testing.assert_array_equal(sampled.predict(pool, task_type="GPU"), ordinary.predict(pool, task_type="GPU"))


@pytest.mark.parametrize("rsm", [.001, .01])
def test_nonbinary_rsm_threshold_is_actionable(tmp_path, rsm):
    pool, _ = problem(tmp_path)
    with pytest.raises(CatBoostError, match="Too low rsm"):
        CatBoost(options("PairLogitPairwise", rsm=rsm)).fit(pool)


def test_sampling_only_a_constant_ctr_keeps_the_source_fallback_and_draw_count(tmp_path):
    row = np.arange(128)
    random = np.random.default_rng(593)
    numeric = np.column_stack([random.permutation(row % 4) for _ in range(8)])
    # Every category occurs once, making all training CTRs equal to the prior.
    # The first eight source features occupy one HalfByte pack; the CTR is the
    # next pack. This source seed selects only that constant CTR, without retry.
    x = np.column_stack([numeric.astype(object), [f"unique{value}" for value in row]])
    pool = Pool(x, (row % 4) / 3, group_id=row // 8, cat_features=[8])
    pool.quantize(border_count=3)
    expected, draws = reference_masks([3] * 8, loss="QueryCrossEntropy", seed=0, trees=1)
    assert expected == [{8}] and draws == 3
    model = CatBoost(options("QueryCrossEntropy", random_seed=0, iterations=1, depth=1)).fit(pool)
    document = exported(model, tmp_path / "constant-ctr.json")
    split, = document["oblivious_trees"][0]["splits"]
    assert split["split_type"] == "OnlineCtr" and split["border"] == .5
    assert int(model.get_metadata()["metal_rsm_host_draw_count"]) == draws
    assert np.isfinite(model.get_leaf_values()).all()
