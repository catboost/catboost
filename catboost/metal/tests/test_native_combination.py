"""Native Combination targets, checked without fitting a CPU reference.

The numerical oracle sums independently differentiated component losses. Public
metrics sum final component metrics, which is deliberately a different operation
from adding the unnormalized values used by the leaf optimizer. Snapshots must
retain every permutation cursor and stochastic Yeti oracle position exactly.
"""

import os
import struct

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool

from test_native_compound_ctrs import (
    CTR_KINDS, check_final_tables, check_readers_and_oracle, exported,
    independent_prediction, options as ctr_options, projections, snapshot_options,
)
from test_native_greedy_api import SCORES, StopAfter, sampler
from test_native_training_modes import (
    check_exact, compound_problem, independent_metric as query_metric,
    metric_history, numeric_problem,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal Combination adapter",
)

PROFILES = {
    "huber": (("RMSE", 1.25), ("Huber:delta=0.8", .75)),
    "query": (("RMSE", 2.), ("QueryRMSE", .6)),
    # The shared metric compatibility check treats the first component as its
    # reference metric. A ranking metric can accompany RMSE; RMSE cannot serve
    # as the reference metric for these ranking losses.
    "pair": (("PairLogit", .75), ("RMSE", 1.25)),
    "yeti": (("YetiRank:permutations=7", .03), ("RMSE", 2.)),
    "multi_yeti": (("YetiRank:permutations=5", .02),
                   ("YetiRank:permutations=9", .03), ("RMSE", 3.)),
}
MODES = ("PlainDP", "PlainFP", "OrderedFP")


@pytest.fixture(autouse=True)
def require_gpu_for_every_fit(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def loss_description(components):
    # A component can contain a colon and one parameter. Semicolons delimit
    # the outer Combination parameters, so none are hidden inside components.
    return "Combination:" + ";".join(
        f"loss{i}={loss};weight{i}={weight}" for i, (loss, weight) in enumerate(components))


def options(profile="huber", mode="PlainDP", count=4, **extra):
    components = PROFILES[profile] if isinstance(profile, str) else profile
    config = dict(
        task_type="GPU", loss_function=loss_description(components),
        boosting_type="Ordered" if mode == "OrderedFP" else "Plain",
        data_partition="DocParallel" if mode == "PlainDP" else "FeatureParallel",
        grow_policy="SymmetricTree", iterations=6, depth=3, learning_rate=.15,
        l2_leaf_reg=2, random_seed=967, border_count=16,
        bootstrap_type="No", random_strength=0, score_function="Cosine",
        leaf_estimation_method="Newton", leaf_estimation_iterations=3,
        leaf_estimation_backtracking="No", fold_size_loss_normalization=False,
        permutation_count=count, has_time=count == 1, metric_period=1,
        verbose=False, allow_writing_files=False,
    )
    if mode == "OrderedFP":
        config.update(min_fold_size=8, fold_len_multiplier=1.7, fold_permutation_block=3)
    return config | extra


def fit(config, pool, **kwargs):
    return CatBoost(config).fit(pool, **kwargs)


def problem(profile):
    return numeric_problem("PairLogit" if profile == "pair" else "QuerySoftMax")


def component_metric(loss, raw, target, pool_options):
    raw, target = np.asarray(raw, float), np.asarray(target, float)
    weight = np.asarray(pool_options.get("weight", np.ones(len(raw))), float)
    residual = target - raw
    if loss == "RMSE":
        return np.sqrt(np.average(residual ** 2, weights=weight))
    if loss.startswith("Huber:"):
        delta = float(loss.partition("delta=")[2])
        value = np.where(np.abs(residual) < delta, .5 * residual ** 2,
                         delta * (np.abs(residual) - .5 * delta))
        return np.average(value, weights=weight)
    return query_metric(loss, raw, target, pool_options)


def combination_metric(components, raw, target, pool_options):
    return sum((-1 if loss.startswith("YetiRank") else 1) * float(np.float32(weight))
               * component_metric(loss, raw, target, pool_options)
               for loss, weight in components if weight)


def check_metric(model, components, target, pool_options, index=-1):
    expected = combination_metric(components, model.get_test_eval(), target, pool_options)
    actual = metric_history(model, "validation", "Combination")[index]
    assert actual == pytest.approx(expected, abs=3e-7, rel=6e-6)


def check_original_mass(model, weights):
    assert model.get_metadata()["metal_backend"] == "METAL"
    at = 0
    for count in model.get_tree_leaf_counts():
        mass = model.get_leaf_weights()[at:at + count]
        assert np.isfinite(mass).all() and np.all(mass >= 0)
        assert mass.sum() == pytest.approx(np.sum(weights, dtype=float), rel=5e-6, abs=5e-6)
        at += count


def readers(model, x, tmp_path):
    expected = model.predict(x, prediction_type="RawFormulaVal", task_type="GPU")
    assert np.isfinite(expected).all()
    for fmt in ("cbm", "json"):
        path = tmp_path / ("combination." + fmt)
        model.save_model(path, format=fmt)
        restored = CatBoost().load_model(path, format=fmt)
        np.testing.assert_array_equal(restored.get_tree_leaf_counts(), model.get_tree_leaf_counts())
        for reader in (model, restored):
            for task in ("CPU", "GPU"):
                np.testing.assert_allclose(
                    reader.predict(x, prediction_type="RawFormulaVal", task_type=task),
                    expected, atol=2e-6, rtol=5e-6)


def analytic_terms(components, target, weights, raw, pool_options):
    """Return weighted row derivatives and weak GradientAt masses.

    Pair edges retain their literal direction, multiplicity, and weight. Their
    component masses are incident pair weights; outer leaf masses remain the
    original object weights. CUDA accumulates queries before pointwise losses.
    """
    target, weights, raw = (np.asarray(a, float) for a in (target, weights, raw))
    totals = [np.zeros(len(raw), np.float32) for _ in range(3)]
    for loss, coefficient in sorted(components, key=lambda c: not c[0].startswith(("Query", "Pair"))):
        residual = target - raw
        mass = weights
        if loss == "RMSE":
            gradient, hessian = weights * residual, weights
        elif loss.startswith("Huber:"):
            delta = float(loss.partition("delta=")[2])
            gradient = weights * np.clip(residual, -delta, delta)
            hessian = weights * (np.abs(residual) < delta)
        elif loss == "QueryRMSE":
            for group in np.unique(pool_options["group_id"]):
                rows = pool_options["group_id"] == group
                if weights[rows].sum():
                    residual[rows] -= np.average(residual[rows], weights=weights[rows])
            gradient, hessian = weights * residual, weights
        else:
            assert loss == "PairLogit"
            gradient, hessian, mass = (np.zeros(len(raw)) for _ in range(3))
            for (winner, loser), pair_weight in zip(pool_options["pairs"], pool_options["pairs_weight"]):
                probability = 1 / (1 + np.exp(raw[winner] - raw[loser]))
                gradient[winner] += pair_weight * probability
                gradient[loser] -= pair_weight * probability
                curvature = pair_weight * probability * (1 - probability)
                hessian[winner] += curvature
                hessian[loser] += curvature
                mass[winner] += pair_weight
                mass[loser] += pair_weight
        for total, value in zip(totals, (gradient, hessian, mass)):
            total += np.float32(coefficient) * np.asarray(value, np.float32)
    return tuple(np.asarray(value, float) for value in totals)


def root_oracle(components, target, weights, baseline, pool_options, method, iterations, l2=2):
    point = np.float32(0)
    for _ in range(iterations):
        raw = np.float32(np.asarray(baseline, np.float32) + point)
        gradient, hessian, _ = analytic_terms(components, target, weights, raw, pool_options)
        diagonal = (np.sum(weights, dtype=float) if method == "Gradient" else hessian.sum()) + l2
        if diagonal > 0:
            point = np.float32(point + np.float32(gradient.sum() / (diagonal + 1e-20)))
    return float(np.float32(.2) * point)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
def test_native_combination_modes_final_component_metrics_and_standard_readers(tmp_path, profile, mode, count, method):
    x, target, po = problem(profile)
    pool = Pool(x, target, **po)
    auxiliary = "PFound" if "yeti" in profile else "PairLogit" if profile == "pair" else "RMSE"
    config = options(profile, mode, count, leaf_estimation_method=method,
                     score_function="NewtonCosine" if method == "Newton" else "Cosine",
                     custom_metric=[auxiliary])
    model = fit(config, pool, eval_set=pool, use_best_model=False)
    assert model.tree_count_ == config["iterations"]
    # Numeric DocParallel follows CUDA Plain's one-dataset fallback when no
    # CTR needs permutation histories. FeatureParallel retains its task count.
    assert model.get_metadata()["metal_permutations"] == str(1 if mode == "PlainDP" else count)
    for key in ("data_partition", "boosting_type", "leaf_estimation_method"):
        assert model.get_all_params()[key] == config[key]
    raw = np.asarray(model.get_test_eval())
    assert np.isfinite(raw).all() and np.ptp(raw) > 1e-5
    check_metric(model, PROFILES[profile], target, po)
    auxiliary_loss = "YetiRank" if auxiliary == "PFound" else auxiliary
    assert metric_history(model, "validation", auxiliary)[-1] == pytest.approx(
        component_metric(auxiliary_loss, raw, target, po), rel=6e-6, abs=3e-7)
    check_original_mass(model, po["weight"])
    readers(model, x, tmp_path)
    if mode == "PlainDP" and count == 4:
        # Also execute four actual DocParallel histories, using a selected
        # simple CTR. The numeric fallback above must not stand in for P4.
        cx, cy, cpo, future, future_y, future_po = compound_problem(
            "PairLogit" if profile == "pair" else "QueryRMSE")
        categorical_config = ctr_options("Borders", "Plain", 4, complexity=1)
        categorical_config.pop("boost_from_average", None)
        categorical_config.update(config)
        categorical = fit(categorical_config, Pool(cx, cy, **cpo),
                          eval_set=Pool(future, future_y, **future_po), use_best_model=False)
        assert categorical.get_metadata()["metal_permutations"] == "4"
        document = exported(categorical, tmp_path / "doc-parallel-ctr.json")
        selected = projections(document)
        assert selected and all(len(projection) == 1 for projection in selected)
        check_final_tables(document, cx, cy)
        expected = independent_prediction(document, cx, cy, future)
        np.testing.assert_allclose(categorical.predict(future, task_type="GPU"), expected,
                                   atol=2e-6, rtol=5e-6)
        check_metric(categorical, PROFILES[profile], future_y, future_po)
        check_original_mass(categorical, cpo["weight"])
        readers(categorical, future, tmp_path)


@pytest.mark.parametrize("mode", ("PlainDP", "PlainFP"))
@pytest.mark.parametrize("score", SCORES)
def test_native_plain_combination_preserves_all_registered_scalar_scores(mode, score):
    x, target, po = problem("pair")
    pool = Pool(x, target, **po)
    model = fit(options("pair", mode, score_function=score), pool, eval_set=pool, use_best_model=False)
    assert model.get_all_params()["score_function"] == score
    assert np.ptp(model.get_test_eval()) > 1e-5
    check_metric(model, PROFILES["pair"], target, po)
    check_original_mass(model, po["weight"])


@pytest.mark.parametrize("profile", ("huber", "query", "pair"))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
def test_native_combination_depth_zero_matches_independent_weighted_derivatives(profile, mode, count, method):
    target = np.tile([0., 1., .25, .75], 4).astype(np.float32)
    weights = np.tile([1., 2., 3., 4.], 4).astype(np.float32)
    baseline = np.tile([-.25, .5, .25, -.5], 4).astype(np.float32)
    x = np.arange(len(target), dtype=np.float32)[:, None]
    po = dict(group_id=np.arange(len(target), dtype=np.uint64) // 2, weight=weights, baseline=baseline)
    if profile == "pair":
        po["pairs"] = np.array([(a, a + 1) for a in range(0, 16, 4)] * 2
                               + [(a + 2, a + 3) for a in range(0, 16, 4)]
                               + [(a + 3, a + 2) for a in range(0, 16, 4)], np.uint32)
        po["pairs_weight"] = np.repeat([2., .5, 3., 0.], 4).astype(np.float32)
    pool = Pool(x, target, **po)
    iterations = 1 if count == 1 else 3
    model = fit(options(profile, mode, count, depth=0, iterations=1, learning_rate=.2,
                        leaf_estimation_method=method, leaf_estimation_iterations=iterations),
                pool, eval_set=pool, use_best_model=False)
    expected = root_oracle(PROFILES[profile], target, weights, baseline, po, method, iterations)
    assert model.get_tree_leaf_counts().tolist() == [1]
    assert model.get_scale_and_bias() == (1., 0.)
    np.testing.assert_allclose(model.get_leaf_values(), [expected], rtol=6e-6, atol=3e-7)
    np.testing.assert_allclose(model.get_test_eval(), baseline + expected, rtol=6e-6, atol=3e-7)
    check_original_mass(model, weights)
    check_metric(model, PROFILES[profile], target, po)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
def test_native_combination_signed_yeti_curvature_has_zero_newton_direction(mode, count, method):
    # In a two-row query with relevances0/1, every classic Yeti permutation
    # gives incident mass .15 to both rows and opposite gradients. The combined
    # root has G=.25*8=2 and H=16*(.25-10*.15)=-20, independent of RNG seed.
    components = (("YetiRank:permutations=7", 10.), ("RMSE", .25))
    target = np.tile([0., 1.], 8).astype(np.float32)
    x = np.arange(len(target), dtype=np.float32)[:, None]
    po = dict(group_id=np.arange(len(target), dtype=np.uint64) // 2, weight=np.ones(len(target), np.float32))
    pool = Pool(x, target, **po)
    model = fit(options(components, mode, count, depth=0, iterations=1, learning_rate=.2,
                        leaf_estimation_method=method, leaf_estimation_iterations=1),
                pool, eval_set=pool, use_best_model=False)
    expected = 0 if method == "Newton" else .2 * 2 / (16 + 2)
    np.testing.assert_allclose(model.get_leaf_values(), [expected], rtol=6e-6, atol=2e-7)
    if method == "Newton":
        np.testing.assert_array_equal(model.get_leaf_values(), [0.])
    else:
        assert model.get_leaf_values()[0] > 0  # Combination must not center this as standalone Yeti.
    check_original_mass(model, po["weight"])
    check_metric(model, components, target, po)


@pytest.mark.parametrize("score", ("Cosine", "NewtonCosine"))
@pytest.mark.parametrize("sampling", ("No", "Bernoulli", "Bayesian"))
def test_native_combination_doc_parallel_simple_exports_sampled_component_weak_statistics(score, sampling):
    x, target, po = problem("huber")
    pool = Pool(x, target, **po)
    config = options("huber", "PlainDP", 1, depth=0, iterations=1, learning_rate=.2,
                     leaf_estimation_method="Simple", leaf_estimation_iterations=1,
                     score_function=score, **sampler(sampling))
    model = fit(config, pool, eval_set=pool, use_best_model=False)
    gradient, hessian, weak_mass = analytic_terms(PROFILES["huber"], target, po["weight"], np.zeros(len(x)), po)
    full_mass = (hessian if score == "NewtonCosine" else weak_mass).sum()
    if sampling == "No":
        expected = .2 * gradient.sum() / (full_mass + config["l2_leaf_reg"])
        np.testing.assert_allclose(model.get_leaf_values(), [expected], rtol=6e-6, atol=3e-7)
        np.testing.assert_allclose(model.get_leaf_weights(), [full_mass], rtol=6e-6, atol=3e-6)
    else:
        assert not np.isclose(model.get_leaf_weights()[0], full_mass, rtol=1e-5)
        unsampled_config = {key: value for key, value in config.items()
                            if key not in ("bagging_temperature", "subsample")}
        unsampled = fit(unsampled_config | sampler("No"), pool, eval_set=pool, use_best_model=False)
        assert not np.array_equal(model.get_leaf_values(), unsampled.get_leaf_values())
    check_metric(model, PROFILES["huber"], target, po)


@pytest.mark.parametrize("mode", ("PlainFP", "OrderedFP"))
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("sampling", ("No", "Bernoulli"))
def test_native_combination_feature_parallel_simple_is_one_original_weight_gradient_step(mode, count, sampling):
    x, target, po = problem("huber")
    pool = Pool(x, target, **po)
    config = options("huber", mode, count, depth=0, iterations=1, learning_rate=.2,
                     leaf_estimation_method="Simple", leaf_estimation_iterations=1, **sampler(sampling))
    simple = fit(config, pool, eval_set=pool, use_best_model=False)
    gradient = fit(config | dict(leaf_estimation_method="Gradient"), pool, eval_set=pool, use_best_model=False)
    check_exact(simple, gradient, x)
    expected = root_oracle(PROFILES["huber"], target, po["weight"], np.zeros(len(x)), po, "Gradient", 1)
    np.testing.assert_allclose(simple.get_leaf_values(), [expected], rtol=6e-6, atol=3e-7)
    check_original_mass(simple, po["weight"])


@pytest.mark.parametrize("profile,sampling", (
    ("huber", "Bernoulli"), ("pair", "Bayesian"),
    ("multi_yeti", "Bernoulli"), ("multi_yeti", "Bayesian"),
    ("multi_yeti", "Poisson"), ("multi_yeti", "MVS"),
))
@pytest.mark.parametrize("mode", MODES)
def test_native_combination_exact_interrupted_completed_extended_snapshots(tmp_path, profile, mode, sampling):
    x, target, po = problem(profile)
    pool = Pool(x, target, **po)
    config = options(profile, mode, random_strength=.4, **sampler(sampling))
    direct = fit(config, pool, eval_set=pool, use_best_model=False)
    saved = snapshot_options(config, tmp_path)
    callback = StopAfter(2)
    partial = fit(saved, pool, eval_set=pool, use_best_model=False, callbacks=[callback])
    assert partial.tree_count_ == 2 and callback.iterations == [1, 2]
    assert (tmp_path / saved["snapshot_file"]).is_file()
    resumed = fit(saved, pool, eval_set=pool, use_best_model=False)
    check_exact(resumed, direct, x)
    check_exact(fit(saved, pool, eval_set=pool, use_best_model=False), direct, x)
    extended = fit(saved | dict(iterations=8), pool, eval_set=pool, use_best_model=False)
    longer = fit(config | dict(iterations=8), pool, eval_set=pool, use_best_model=False)
    check_exact(extended, longer, x)
    check_metric(extended, PROFILES[profile], target, po)
    readers(extended, x, tmp_path)
    changed = target.copy()
    changed[0] += .03125
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot"):
        fit(saved | dict(iterations=8), Pool(x, changed, **po), eval_set=pool, use_best_model=False)


@pytest.mark.parametrize("profile", ("yeti", "multi_yeti"))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("backtracking", ("AnyImprovement", "Armijo"))
def test_native_combination_yeti_trial_oracles_resume_exactly(tmp_path, profile, mode, backtracking):
    x, target, po = problem(profile)
    pool = Pool(x, target, **po)
    config = options(profile, mode, iterations=5, leaf_estimation_iterations=4,
                     leaf_estimation_backtracking=backtracking, bootstrap_type="Bayesian", bagging_temperature=.7)
    direct = fit(config, pool, eval_set=pool, use_best_model=False)
    saved = snapshot_options(config, tmp_path)
    fit(saved, pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(2)])
    resumed = fit(saved, pool, eval_set=pool, use_best_model=False)
    check_exact(resumed, direct, x)
    check_exact(fit(saved, pool, eval_set=pool, use_best_model=False), direct, x)
    check_metric(resumed, PROFILES[profile], target, po)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("completed", (False, True))
def test_native_combination_rejects_corrupt_stochastic_draw_count_even_when_complete(tmp_path, mode, completed):
    x, target, po = problem("multi_yeti")
    pool = Pool(x, target, **po)
    saved = snapshot_options(options("multi_yeti", mode, iterations=4), tmp_path)
    args = {} if completed else dict(callbacks=[StopAfter(2)])
    model = fit(saved, pool, eval_set=pool, use_best_model=False, **args)
    assert model.tree_count_ == (4 if completed else 2)
    path = tmp_path / saved["snapshot_file"]
    raw = bytearray(path.read_bytes())
    # DP adds a tagged Combination RNG record before its optional model-history
    # tail; FP stores RNG metadata in the terminal common Ordered slot.
    # TProgressHelper logs its MD5 but appends no checksum to these bytes.
    tag = b"Metal Combination target random v1"
    assert (tag in raw) == (mode == "PlainDP")
    if mode == "PlainDP":
        from native_snapshot_tail import stochastic_tail
        offset, expected_state = stochastic_tail(raw, tag, trees=model.tree_count_,
            permutations=int(model.get_metadata()["metal_permutations"]), leaf_capacity=1 << saved["depth"], dimension=1)
        position = offset + len(tag)
        assert stochastic_tail(raw[:position + 13], tag) == (offset, expected_state)
    else:
        position = len(raw) - struct.calcsize("<QIB")
    draws, iterations, initialized = struct.unpack_from("<QIB", raw, position)
    # DocParallel creates its bootstrap cache only for a sampled bootstrap;
    # FeatureParallel initializes the CUDA-compatible cache even for No.
    assert draws > 0 and iterations == model.tree_count_
    assert initialized == int(mode != "PlainDP")
    struct.pack_into("<Q", raw, position, 0)
    path.write_bytes(raw)
    with pytest.raises(CatBoostError, match="(?i)draw count|random.*snapshot|snapshot.*random"):
        fit(saved, pool, eval_set=pool, use_best_model=False)
    assert path.read_bytes() == raw


def compound_config(profile, mode, count=4, kind="Borders", history="Group", **extra):
    config = ctr_options(kind, "Ordered" if mode == "OrderedFP" else "Plain", count)
    config.pop("boost_from_average", None)
    config.update(options(profile, mode, count), iterations=8, depth=4,
                  max_ctr_complexity=2, ctr_history_unit=history)
    return config | extra


@pytest.mark.parametrize("profile", ("query", "multi_yeti"))
@pytest.mark.parametrize("mode", ("PlainFP", "OrderedFP"))
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("kind", CTR_KINDS)
def test_native_combination_selects_compounds_and_predicts_unseen_keys(tmp_path, profile, mode, count, kind):
    x, target, po, future, future_target, future_po = compound_problem("QueryRMSE")
    pool = Pool(x, target, **po)
    if count == 4:
        pool.quantize(border_count=16)
    evaluation = Pool(future, future_target, **future_po)
    config = compound_config(profile, mode, count, kind, "Group" if count == 4 else "Sample")
    model = fit(config, pool, eval_set=evaluation, use_best_model=False)
    check_readers_and_oracle(model, x, target, future, tmp_path)
    check_metric(model, PROFILES[profile], future_target, future_po)
    check_original_mass(model, po["weight"])


@pytest.mark.parametrize("mode", ("PlainFP", "OrderedFP"))
def test_native_combination_complexity_three_is_actually_selected(tmp_path, mode):
    x, target, po, future, future_target, future_po = compound_problem("QueryRMSE", 3)
    config = compound_config("multi_yeti", mode, max_ctr_complexity=3, iterations=12, depth=5)
    model = fit(config, Pool(x, target, **po), eval_set=Pool(future, future_target, **future_po), use_best_model=False)
    check_readers_and_oracle(model, x, target, future, tmp_path, minimum_complexity=3)
    check_metric(model, PROFILES["multi_yeti"], future_target, future_po)


@pytest.mark.parametrize("profile", ("pair", "multi_yeti"))
@pytest.mark.parametrize("mode", ("PlainFP", "OrderedFP"))
def test_native_combination_compound_registry_survives_exact_snapshot_extension(tmp_path, profile, mode):
    x, target, po, future, future_target, future_po = compound_problem("PairLogit" if profile == "pair" else "QueryRMSE")
    pool, evaluation = Pool(x, target, **po), Pool(future, future_target, **future_po)
    pool.quantize(border_count=16)
    config = compound_config(profile, mode, kind="FloatTargetMeanValue",
                             bootstrap_type="Bernoulli", subsample=.8, random_strength=.3)
    saved = snapshot_options(config, tmp_path)
    partial = fit(saved, pool, eval_set=evaluation, use_best_model=False, callbacks=[StopAfter(3)])
    assert partial.tree_count_ == 3
    assert any(len(p) > 1 for p in projections(exported(partial, tmp_path / "partial.json")))
    resumed = fit(saved, pool, eval_set=evaluation, use_best_model=False)
    direct = fit(config, pool, eval_set=evaluation, use_best_model=False)
    check_exact(resumed, direct, future)
    extended = fit(saved | dict(iterations=10), pool, eval_set=evaluation, use_best_model=False)
    longer = fit(config | dict(iterations=10), pool, eval_set=evaluation, use_best_model=False)
    check_exact(extended, longer, future)
    check_readers_and_oracle(extended, x, target, future, tmp_path)
    check_metric(extended, PROFILES[profile], future_target, future_po)


@pytest.mark.parametrize("profile", ("pair", "multi_yeti"))
@pytest.mark.parametrize("mode", MODES)
def test_native_combination_initial_model_baseline_and_snapshot_are_cumulative(tmp_path, profile, mode):
    x, target, po = problem(profile)
    pool = Pool(x, target, **po)
    config = options(profile, mode, iterations=4, random_strength=.3,
                     bootstrap_type="Bernoulli", subsample=.8)
    initial = fit(config | dict(iterations=2), pool)
    baseline = np.linspace(-.2, .3, len(x), dtype=np.float32)
    pool.set_baseline(baseline)
    args = dict(init_model=initial, eval_set=pool, use_best_model=False)
    direct = fit(config, pool, **args)
    saved = snapshot_options(config, tmp_path)
    callback = StopAfter(2)
    partial = fit(saved, pool, callbacks=[callback], **args)
    assert partial.tree_count_ == 4 and callback.iterations == [1, 2]
    resumed = fit(saved, pool, **args)
    assert resumed.tree_count_ == 6
    check_exact(resumed, direct, x)
    np.testing.assert_allclose(resumed.get_test_eval(), resumed.predict(x, task_type="GPU") + baseline,
                               rtol=5e-6, atol=2e-6)
    check_metric(resumed, PROFILES[profile], target, po)
    readers(resumed, x, tmp_path)


@pytest.mark.parametrize("profile", ("huber", "multi_yeti"))
@pytest.mark.parametrize("mode", MODES)
def test_native_combination_best_default_metric_and_terminal_snapshot_recover(tmp_path, profile, mode):
    x, target, po = problem(profile)
    pool = Pool(x, target, **po)
    reversed_target = np.float32(target.max() + target.min() - target)
    evaluation = Pool(x, reversed_target, **po)
    config = options(profile, mode, iterations=15, learning_rate=.4)
    args = dict(eval_set=evaluation, use_best_model=True)
    direct = fit(config, pool, **args)
    history = metric_history(direct, "validation", "Combination")
    best = int(np.argmin(history))
    assert direct.get_best_iteration() == best
    assert direct.tree_count_ == best + 1 < config["iterations"]
    check_metric(direct, PROFILES[profile], reversed_target, po, index=best)
    saved = snapshot_options(config, tmp_path)
    fit(saved, pool, callbacks=[StopAfter(2)], **args)
    resumed = fit(saved, pool, **args)
    check_exact(resumed, direct, x)
    early = fit(config, pool, early_stopping_rounds=2, **args)
    assert len(metric_history(early, "validation", "Combination")) < config["iterations"]
    early_saved = snapshot_options(config, tmp_path / "early")
    fit(early_saved, pool, callbacks=[StopAfter(1)], early_stopping_rounds=2, **args)
    check_exact(fit(early_saved, pool, early_stopping_rounds=2, **args), early, x)
    check_exact(fit(early_saved, pool, early_stopping_rounds=2, **args), early, x)
    readers(resumed, x, tmp_path)


@pytest.mark.parametrize("mode", MODES)
def test_native_combination_zero_component_is_omitted_and_effective_weights_are_combined_once(mode):
    x, target, po = problem("query")
    group_weights = (.5 + po["group_id"] / 32).astype(np.float32)
    separate = Pool(x, target, group_id=po["group_id"], group_weight=group_weights)
    separate.set_weight(po["weight"])
    combined_po = po | dict(weight=np.float32(po["weight"] * group_weights))
    combined = Pool(x, target, **combined_po)
    config = options("query", mode, iterations=4)
    direct = fit(config, combined, eval_set=combined, use_best_model=False)
    check_exact(fit(config, separate, eval_set=separate, use_best_model=False), direct, x)
    # A skipped Yeti component must consume no oracle seeds and add no PFound.
    zero = PROFILES["query"] + (("YetiRank", 0.),)
    skipped = fit(config | dict(loss_function=loss_description(zero)), combined, eval_set=combined, use_best_model=False)
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(skipped, method)(), getattr(direct, method)())
    np.testing.assert_array_equal(metric_history(skipped, "validation", "Combination"),
                                  metric_history(direct, "validation", "Combination"))
    check_metric(skipped, PROFILES["query"], target, combined_po)


@pytest.mark.parametrize("extra,message", (
    ({"grow_policy": "Depthwise"}, "(?i)symmetric|Combination"),
    ({"grow_policy": "Lossguide"}, "(?i)symmetric|Combination"),
    ({"grow_policy": "Region"}, "(?i)symmetric|Combination"),
    ({"leaf_estimation_method": "Exact"}, "(?i)Exact|leaf estimation|Newton or Gradient"),
    ({"sampling_unit": "Group"}, "(?i)sampling_unit|YetiRankPairwise|object bootstrap"),
    ({"max_ctr_complexity": 2}, "(?i)FeatureParallel"),
))
def test_native_combination_rejects_unregistered_training_boundaries(extra, message):
    x, target, po = problem("query")
    with pytest.raises(CatBoostError, match=message):
        fit(options("query", "PlainDP", iterations=1, **extra), Pool(x, target, **po))


@pytest.mark.parametrize("component", ("QueryCrossEntropy", "PairLogitPairwise", "YetiRankPairwise", "MultiRMSE"))
def test_native_combination_rejects_components_without_cuda_diagonal_registration(component):
    x, target, po = problem("pair")
    with pytest.raises(CatBoostError, match="(?i)unsupported|compatible|multi|Combination"):
        fit(options((("RMSE", 2.), (component, .5)), iterations=1), Pool(x, target, **po))


@pytest.mark.parametrize("components,message", (
    ((("RMSE", 1.), ("Huber:delta=0.8", -1.)), "(?i)weight|Combination"),
    ((("RMSE", 0.), ("QueryRMSE", 0.)), "(?i)non.zero|weight|Combination"),
    ((("RMSE", 1.), ("Huber:delta=-0.8", 1.)), "(?i)delta|Huber|nonnegative"),
    ((("YetiRank:permutations=0", 1.), ("RMSE", 1.)), "(?i)YetiRank|permutations"),
    ((("YetiRank:mode=NDCG", 1.), ("RMSE", 1.)), "(?i)Classic|classic|YetiRank"),
))
def test_native_combination_validates_component_parameters(components, message):
    x, target, po = problem("query")
    with pytest.raises(CatBoostError, match=message):
        fit(options(components, iterations=1), Pool(x, target, **po))


@pytest.mark.parametrize("component,target_value,message", (
    ("Poisson", -.5, "(?i)nonnegative|non-negative|greater.*equal.*0|negative target"),
    ("Tweedie:variance_power=1.3", -.5, "(?i)nonnegative|non-negative|greater.*equal.*0|negative target"),
    ("QuerySoftMax", 0., "(?i)positive|weighted target|sum.*target"),
))
def test_native_combination_checks_every_component_target_domain(component, target_value, message):
    x, _, po = problem("query")
    target = np.full(len(x), target_value, np.float32)
    components = ((component, .5), ("RMSE", 1.)) if component == "QuerySoftMax" else (("RMSE", 1.), (component, .5))
    config = options(components, iterations=1, allow_const_label=True)
    with pytest.raises(CatBoostError, match=message):
        fit(config, Pool(x, target, **po))


def test_native_combination_cross_entropy_does_not_binarize_invalid_raw_labels():
    x, target, po = problem("query")
    target = target.copy()
    target[3] = 1.5
    components = (("CrossEntropy", 1.), ("Logloss:border=0.6", .5))
    with pytest.raises(CatBoostError, match="(?i)\\[0.?[,;].?1\\]|range|CrossEntropy|less.*equal.*1"):
        fit(options(components, iterations=1), Pool(x, target, **po))


@pytest.mark.parametrize("mode", ("PlainFP", "OrderedFP"))
def test_native_ordered_combination_partition_and_score_boundaries(mode):
    x, target, po = problem("query")
    invalid = dict(boosting_type="Ordered", data_partition="DocParallel") if mode == "PlainFP" else dict(score_function="L2")
    with pytest.raises(CatBoostError, match="(?i)Ordered|FeatureParallel|Cosine"):
        fit(options("query", mode, iterations=1, **invalid), Pool(x, target, **po))
