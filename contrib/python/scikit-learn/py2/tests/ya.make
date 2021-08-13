PY2TEST()



SIZE(MEDIUM)

TIMEOUT(600)
# With 32 cores:
# real    5m37.704s
# user    122m53.632s

FORK_TESTS()

PEERDIR(
    contrib/python/nose
    contrib/python/Pillow
    contrib/python/pandas
    contrib/python/scikit-learn
)

DATA(
    arcadia/contrib/python/scikit-learn/py2/sklearn/datasets
)

SRCDIR(contrib/python/scikit-learn/py2)

NO_LINT()

TEST_SRCS(
    # FIXME: These tests should not fail!
    #sklearn/neighbors/tests/test_ball_tree.py
    #sklearn/neighbors/tests/test_kd_tree.py
    #sklearn/neighbors/tests/test_kde.py

    # This was not adapted and is not relevant.
    #sklearn/tests/test_check_build.py

    # Depend on external datasets.
    #sklearn/datasets/tests/test_20news.py
    #sklearn/datasets/tests/test_covtype.py
    #sklearn/datasets/tests/test_kddcup99.py
    #sklearn/datasets/tests/test_mldata.py
    #sklearn/datasets/tests/test_rcv1.py

    # Depends on disabled scipy/misc/pilutil.py.
    #sklearn/datasets/tests/test_lfw.py

    # Depends on contrib/python/scipy/scipy/misc data files.
    #sklearn/feature_extraction/tests/test_image.py

    sklearn/cluster/tests/__init__.py
    sklearn/cluster/tests/common.py
    sklearn/cluster/tests/test_affinity_propagation.py
    sklearn/cluster/tests/test_bicluster.py
    sklearn/cluster/tests/test_birch.py
    sklearn/cluster/tests/test_dbscan.py
    sklearn/cluster/tests/test_hierarchical.py
    sklearn/cluster/tests/test_k_means.py
    sklearn/cluster/tests/test_mean_shift.py
    sklearn/cluster/tests/test_spectral.py
    sklearn/covariance/tests/__init__.py
    sklearn/covariance/tests/test_covariance.py
    sklearn/covariance/tests/test_graph_lasso.py
    sklearn/covariance/tests/test_robust_covariance.py
    sklearn/cross_decomposition/tests/__init__.py
    sklearn/cross_decomposition/tests/test_pls.py
    sklearn/datasets/tests/__init__.py
    sklearn/datasets/tests/test_base.py
    sklearn/datasets/tests/test_samples_generator.py
    sklearn/datasets/tests/test_svmlight_format.py
    sklearn/decomposition/tests/__init__.py
    sklearn/decomposition/tests/test_dict_learning.py
    sklearn/decomposition/tests/test_factor_analysis.py
    sklearn/decomposition/tests/test_fastica.py
    sklearn/decomposition/tests/test_incremental_pca.py
    sklearn/decomposition/tests/test_kernel_pca.py
    sklearn/decomposition/tests/test_nmf.py
    sklearn/decomposition/tests/test_online_lda.py
    sklearn/decomposition/tests/test_pca.py
    sklearn/decomposition/tests/test_sparse_pca.py
    sklearn/decomposition/tests/test_truncated_svd.py
    sklearn/ensemble/tests/__init__.py
    sklearn/ensemble/tests/test_bagging.py
    sklearn/ensemble/tests/test_base.py
    sklearn/ensemble/tests/test_forest.py
    sklearn/ensemble/tests/test_gradient_boosting.py
    sklearn/ensemble/tests/test_gradient_boosting_loss_functions.py
    sklearn/ensemble/tests/test_iforest.py
    sklearn/ensemble/tests/test_partial_dependence.py
    sklearn/ensemble/tests/test_voting_classifier.py
    sklearn/ensemble/tests/test_weight_boosting.py
    sklearn/feature_extraction/tests/__init__.py
    sklearn/feature_extraction/tests/test_dict_vectorizer.py
    sklearn/feature_extraction/tests/test_feature_hasher.py
    sklearn/feature_extraction/tests/test_text.py
    sklearn/feature_selection/tests/__init__.py
    sklearn/feature_selection/tests/test_base.py
    sklearn/feature_selection/tests/test_chi2.py
    sklearn/feature_selection/tests/test_feature_select.py
    sklearn/feature_selection/tests/test_from_model.py
    sklearn/feature_selection/tests/test_mutual_info.py
    sklearn/feature_selection/tests/test_rfe.py
    sklearn/feature_selection/tests/test_variance_threshold.py
    sklearn/gaussian_process/tests/__init__.py
    sklearn/gaussian_process/tests/test_gaussian_process.py
    sklearn/gaussian_process/tests/test_gpc.py
    sklearn/gaussian_process/tests/test_gpr.py
    sklearn/gaussian_process/tests/test_kernels.py
    sklearn/linear_model/tests/__init__.py
    sklearn/linear_model/tests/test_base.py
    sklearn/linear_model/tests/test_bayes.py
    sklearn/linear_model/tests/test_coordinate_descent.py
    sklearn/linear_model/tests/test_huber.py
    sklearn/linear_model/tests/test_least_angle.py
    sklearn/linear_model/tests/test_logistic.py
    sklearn/linear_model/tests/test_omp.py
    sklearn/linear_model/tests/test_passive_aggressive.py
    sklearn/linear_model/tests/test_perceptron.py
    sklearn/linear_model/tests/test_randomized_l1.py
    sklearn/linear_model/tests/test_ransac.py
    sklearn/linear_model/tests/test_ridge.py
    sklearn/linear_model/tests/test_sag.py
    sklearn/linear_model/tests/test_sgd.py
    sklearn/linear_model/tests/test_sparse_coordinate_descent.py
    sklearn/linear_model/tests/test_theil_sen.py
    sklearn/manifold/tests/__init__.py
    sklearn/manifold/tests/test_isomap.py
    sklearn/manifold/tests/test_locally_linear.py
    sklearn/manifold/tests/test_mds.py
    sklearn/manifold/tests/test_spectral_embedding.py
    sklearn/manifold/tests/test_t_sne.py
    sklearn/metrics/cluster/tests/__init__.py
    sklearn/metrics/cluster/tests/test_bicluster.py
    sklearn/metrics/cluster/tests/test_supervised.py
    sklearn/metrics/cluster/tests/test_unsupervised.py
    sklearn/metrics/tests/__init__.py
    sklearn/metrics/tests/test_classification.py
    sklearn/metrics/tests/test_common.py
    sklearn/metrics/tests/test_pairwise.py
    sklearn/metrics/tests/test_ranking.py
    sklearn/metrics/tests/test_regression.py
    sklearn/metrics/tests/test_score_objects.py
    sklearn/mixture/tests/__init__.py
    sklearn/mixture/tests/test_bayesian_mixture.py
    sklearn/mixture/tests/test_dpgmm.py
    sklearn/mixture/tests/test_gaussian_mixture.py
    sklearn/mixture/tests/test_gmm.py
    sklearn/model_selection/tests/__init__.py
    sklearn/model_selection/tests/common.py
    sklearn/model_selection/tests/test_search.py
    sklearn/model_selection/tests/test_split.py
    sklearn/model_selection/tests/test_validation.py
    sklearn/neighbors/tests/__init__.py
    sklearn/neighbors/tests/test_approximate.py
    sklearn/neighbors/tests/test_dist_metrics.py
    sklearn/neighbors/tests/test_nearest_centroid.py
    sklearn/neighbors/tests/test_neighbors.py
    sklearn/neural_network/tests/__init__.py
    sklearn/neural_network/tests/test_mlp.py
    sklearn/neural_network/tests/test_rbm.py
    sklearn/neural_network/tests/test_stochastic_optimizers.py
    sklearn/preprocessing/tests/__init__.py
    sklearn/preprocessing/tests/test_data.py
    sklearn/preprocessing/tests/test_function_transformer.py
    sklearn/preprocessing/tests/test_imputation.py
    sklearn/preprocessing/tests/test_label.py
    sklearn/semi_supervised/tests/__init__.py
    sklearn/semi_supervised/tests/test_label_propagation.py
    sklearn/svm/tests/__init__.py
    sklearn/svm/tests/test_bounds.py
    sklearn/svm/tests/test_sparse.py
    sklearn/svm/tests/test_svm.py
    sklearn/tests/__init__.py
    sklearn/tests/test_base.py
    sklearn/tests/test_calibration.py
    sklearn/tests/test_common.py
    sklearn/tests/test_cross_validation.py
    sklearn/tests/test_discriminant_analysis.py
    sklearn/tests/test_dummy.py
    sklearn/tests/test_grid_search.py
    sklearn/tests/test_init.py
    sklearn/tests/test_isotonic.py
    sklearn/tests/test_kernel_approximation.py
    sklearn/tests/test_kernel_ridge.py
    sklearn/tests/test_learning_curve.py
    sklearn/tests/test_metaestimators.py
    sklearn/tests/test_multiclass.py
    sklearn/tests/test_multioutput.py
    sklearn/tests/test_naive_bayes.py
    sklearn/tests/test_pipeline.py
    sklearn/tests/test_random_projection.py
    sklearn/tree/tests/__init__.py
    sklearn/tree/tests/test_export.py
    sklearn/tree/tests/test_tree.py
    sklearn/utils/sparsetools/tests/__init__.py
    sklearn/utils/sparsetools/tests/test_traversal.py
    sklearn/utils/tests/__init__.py
    sklearn/utils/tests/test_bench.py
    sklearn/utils/tests/test_class_weight.py
    sklearn/utils/tests/test_estimator_checks.py
    sklearn/utils/tests/test_extmath.py
    sklearn/utils/tests/test_fast_dict.py
    sklearn/utils/tests/test_fixes.py
    sklearn/utils/tests/test_graph.py
    sklearn/utils/tests/test_linear_assignment.py
    sklearn/utils/tests/test_metaestimators.py
    sklearn/utils/tests/test_multiclass.py
    sklearn/utils/tests/test_murmurhash.py
    sklearn/utils/tests/test_optimize.py
    sklearn/utils/tests/test_random.py
    sklearn/utils/tests/test_seq_dataset.py
    sklearn/utils/tests/test_shortest_path.py
    sklearn/utils/tests/test_sparsefuncs.py
    sklearn/utils/tests/test_stats.py
    sklearn/utils/tests/test_testing.py
    sklearn/utils/tests/test_utils.py
    sklearn/utils/tests/test_validation.py
)

END()
