"""Self-check the Ordered equation oracle without importing the Metal backend."""

import unittest

import numpy as np

from cuda_ordered_reference import numeric_folds, objective_terms, train_reference


def fixture(rows=33, **options):
    rng = np.random.default_rng(451)
    bins = rng.integers(0, 5, (3, rows), dtype=np.uint8)
    targets = (.3 * bins[0] - .7 * bins[1] + .15 * rng.normal(size=rows)).astype(np.float32)
    settings = dict(iterations=3, depth=2, learning_rate=.13, l2_leaf_reg=3, bias=.2)
    settings.update(options)
    return bins, targets, np.repeat(np.arange(3), 4), np.tile(np.arange(4), 3), settings


class OrderedReferenceTests(unittest.TestCase):
    def test_scalar_objective_equations_at_hand_calculated_boundaries(self):
        cases = [
            ("Huber", 1, [2, 0, -2], [1.5, 0, 1.5], [1, 0, -1], [0, 1, 0]),
            ("Expectile", .25, [2, 0, -2], [1, 0, 3], [1, 0, -3], [.5, 1.5, 1.5]),
            ("Lq", 1, [2, 0, -2], [2, 0, 2], [1, -1, -1], [1, 1, 1]),
            ("Lq", 2, [2, 0, -2], [4, 0, 4], [4, 0, -4], [2, 2, 2]),
            ("Quantile", .25, [2, 0, -2], [.5, 0, 1.5], [.25, -.75, -.75], [0, 0, 0]),
            ("MAE", None, [2, 0, -2], [2, 0, 2], [.5, -.5, -.5], [0, 0, 0]),
            ("MAPE", None, [2, 0, -2], [1, 0, 1], [.5, -1, -.5], [0, 0, 0]),
            ("Poisson", None, [2, 0, 3], [1, 1, 1], [1, -1, 2], [1, 1, 1]),
            ("Tweedie", 1.5, [2, 0, 3], [6, 2, 8], [1, -1, 2], [1.5, .5, 2]),
            ("LogLinQuantile", .25, [2, 1, 0], [.25, 0, .75], [.25, -.75, -.75], [0, 0, 0]),
        ]
        for objective, parameter, targets, loss, gradient, curvature in cases:
            with self.subTest(objective=objective, parameter=parameter):
                actual = objective_terms(targets, np.zeros(3), objective, parameter)
                np.testing.assert_array_equal(actual, [loss, gradient, curvature])

    def test_numeric_next_row_boundaries(self):
        for rows, expected in (
            (4, [(2, 4)]), (9, [(2, 5), (5, 9)]),
            (33, [(2, 5), (5, 11), (11, 23), (23, 33)]),
            (500, [(11, 23), (23, 47), (47, 95), (95, 191), (191, 383), (383, 500)]),
        ):
            with self.subTest(rows=rows):
                np.testing.assert_array_equal(numeric_folds(rows), expected)

    def test_slow_growth_advances_and_large_minimum_cap(self):
        np.testing.assert_array_equal(numeric_folds(9, 1.1),
                                      [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)])
        self.assertEqual(int(numeric_folds(1 << 24, min_fold_size=1)[0, 0]), 65)

    def test_one_leaf_prefix_and_full_model_have_independent_values(self):
        result = train_reference(np.zeros((0, 9), np.uint8), np.arange(9), [], [],
                                 iterations=1, depth=1, learning_rate=.25, l2_leaf_reg=2, bias=0)
        expected = .25 * np.array([1 / 4, 10 / 7, 36 / 11])
        np.testing.assert_allclose(result["task_leaf_values"][0, :, 0], expected, rtol=1e-14)
        np.testing.assert_allclose(result["predictions"], expected[-1], rtol=1e-14)
        for task, (_, end, offset, _) in enumerate(result["folds"]):
            np.testing.assert_allclose(result["cursors"][offset:offset + end], expected[task], rtol=1e-14)
        self.assertEqual(result["depths"].tolist(), [0])
        self.assertEqual(result["leaf_weights"][0, 0], 9)

    def test_normalization_uses_total_prefix_weight_for_leaf_ridge(self):
        targets = np.arange(9, dtype=np.float32)
        weights = np.arange(1, 10, dtype=np.float32)
        result = train_reference(np.zeros((0, 9), np.uint8), targets, [], [],
            iterations=1, depth=0, learning_rate=.25, l2_leaf_reg=2, bias=0,
            sample_weight=weights, fold_size_loss_normalization=True)
        for task, (prefix, _, _, _) in enumerate(result["folds"]):
            expected = .25 * np.dot(targets[:prefix], weights[:prefix]) / (3 * weights[:prefix].sum())
            self.assertAlmostEqual(result["task_leaf_values"][0, task, 0], expected, places=7)

    def test_repeated_leaf_steps_are_unshrunk_until_task_completion(self):
        result = train_reference(np.zeros((0, 9), np.uint8), np.arange(9), [], [],
            iterations=1, depth=0, learning_rate=.25, l2_leaf_reg=2, bias=0,
            leaf_estimation_iterations=4)
        expected = .25 * 4 * (1 - (2 / 11) ** 4)
        self.assertAlmostEqual(result["leaf_values"][0, 0], expected, places=13)

    def test_quality_labels_never_enter_that_folds_leaf_estimation(self):
        common = dict(iterations=3, depth=0, learning_rate=.25, l2_leaf_reg=2, bias=0)
        original = np.arange(9, dtype=np.float32)
        changed = original.copy()
        changed[2:] += 100
        before = train_reference(np.zeros((0, 9), np.uint8), original, [], [], **common)
        after = train_reference(np.zeros((0, 9), np.uint8), changed, [], [], **common)
        np.testing.assert_array_equal(before["task_leaf_values"][:, 0], after["task_leaf_values"][:, 0])
        np.testing.assert_array_equal(before["cursors"][:5], after["cursors"][:5])
        self.assertGreater(abs(before["predictions"][0] - after["predictions"][0]), 10)

    def test_zero_weight_prefix_is_safe_and_unchanged(self):
        weights = np.ones(9, dtype=np.float32)
        weights[:2] = 0
        result = train_reference(np.zeros((0, 9), np.uint8), np.arange(9), [], [],
            iterations=3, depth=0, learning_rate=.25, l2_leaf_reg=0, bias=1,
            sample_weight=weights, fold_size_loss_normalization=True)
        np.testing.assert_array_equal(result["cursors"][:5], np.ones(5))
        self.assertTrue(np.isfinite(result["cursors"]).all())

    def test_all_permutations_update_but_cuda_selector_excludes_last_learning(self):
        bins, targets, features, borders, options = fixture(iterations=6, permutation_count=4)
        result = train_reference(bins, targets, features, borders, **options)
        self.assertEqual(set(result["selected_permutations"]), {0, 1})
        self.assertEqual(set(result["folds"][:, 3]), {0, 1, 2, 3})
        for _, end, offset, _ in result["folds"]:
            self.assertFalse(np.all(result["cursors"][offset:offset + end] == np.float32(.2)))
        # Independent cursor updates can disagree even for the same original row.
        self.assertNotEqual(result["cursors"][0], result["cursors"][5])

    def test_resume_restores_every_fold_and_absolute_selection_iteration(self):
        bins, targets, features, borders, options = fixture(iterations=7, permutation_count=4)
        whole = train_reference(bins, targets, features, borders, **options)
        first = train_reference(bins, targets, features, borders, **dict(options, iterations=3))
        second = train_reference(bins, targets, features, borders,
                                 **dict(options, iterations=4, initial_state=first["state"]))
        for key in ("depths", "split_features", "split_bins", "leaf_values", "leaf_weights", "selected_permutations"):
            np.testing.assert_array_equal(whole[key], np.concatenate([first[key], second[key]]))
        np.testing.assert_array_equal(whole["cursors"], second["cursors"])
        self.assertEqual(second["completed_iterations"], 7)

    def test_init_predictions_follow_original_rows_in_each_permutation(self):
        bins, targets, features, borders, options = fixture(iterations=0, permutation_count=4)
        baseline = np.linspace(-1, 1, targets.size).astype(np.float32)
        result = train_reference(bins, targets, features, borders,
                                 **dict(options, initial_predictions=baseline))
        for _, end, offset, permutation in result["folds"]:
            np.testing.assert_array_equal(result["cursors"][offset:offset + end],
                                          baseline[result["permutations"][permutation, :end]])
        np.testing.assert_array_equal(result["predictions"], baseline)

    def test_binary_objectives_match_for_hard_labels_and_remain_finite(self):
        bins, targets, features, borders, options = fixture(leaf_estimation_iterations=3)
        labels = (targets > np.median(targets)).astype(np.float32)
        for score in ("Cosine", "NewtonCosine"):
            with self.subTest(score=score):
                logloss = train_reference(bins, labels, features, borders,
                    **dict(options, objective="Logloss", score_function=score))
                entropy = train_reference(bins, labels, features, borders,
                    **dict(options, objective="CrossEntropy", score_function=score))
                np.testing.assert_array_equal(logloss["cursors"], entropy["cursors"])
                self.assertTrue(np.isfinite(logloss["loss"]).all())

    def test_gradient_and_newton_match_for_rmse(self):
        bins, targets, features, borders, options = fixture(leaf_estimation_iterations=3)
        newton = train_reference(bins, targets, features, borders, **options)
        gradient = train_reference(bins, targets, features, borders,
                                   **dict(options, leaf_estimation_method="Gradient"))
        np.testing.assert_array_equal(newton["cursors"], gradient["cursors"])

    def test_duplicate_winner_stops_before_readding_split(self):
        bins = np.asarray([[0, 0, 1, 0, 1, 0, 1, 1, 1]], dtype=np.uint8)
        targets = 2 * bins[0].astype(np.float32) - 1
        result = train_reference(bins, targets, [0], [0], iterations=2, depth=4,
                                 learning_rate=.25, l2_leaf_reg=2, bias=0)
        np.testing.assert_array_equal(result["depths"], [1, 1])


if __name__ == "__main__":
    unittest.main()
