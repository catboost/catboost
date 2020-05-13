LIBRARY()



SRCS(
    kernel/pointwise_hist2.cu
    kernel/pointwise_hist2_binary.cu
    kernel/pointwise_hist2_half_byte.cu
    kernel/pointwise_hist2_one_byte_5bit.cu
    kernel/pointwise_hist2_one_byte_6bit.cu
    kernel/pointwise_hist2_one_byte_7bit.cu
    kernel/pointwise_hist2_one_byte_8bit.cu
    kernel/pointwise_hist1.cu
    kernel/pointwise_scores.cu
    kernel/linear_solver.cu
    kernel/pairwise_hist.cu
    kernel/pairwise_hist_binary.cu
    kernel/pairwise_hist_half_byte.cu

    kernel/pairwise_hist_one_byte_5bit.cu
    kernel/pairwise_hist_one_byte_6bit.cu
    kernel/pairwise_hist_one_byte_7bit.cu
    kernel/pairwise_hist_one_byte_8bit_atomics.cu

    kernel/pairwise_hist_one_byte_5bit_one_hot.cu
    kernel/pairwise_hist_one_byte_6bit_one_hot.cu
    kernel/pairwise_hist_one_byte_7bit_one_hot.cu
    kernel/pairwise_hist_one_byte_8bit_atomics_one_hot.cu
    kernel/split_pairwise.cu

    greedy_subsets_searcher/split_points.cpp
    greedy_subsets_searcher/model_builder.cpp
    greedy_subsets_searcher/compute_by_blocks_helper.cpp
    greedy_subsets_searcher/split_properties_helper.cpp
    greedy_subsets_searcher/greedy_search_helper.cpp
    greedy_subsets_searcher/kernel/gather_bins.cu
    greedy_subsets_searcher/kernel/hist.cu
    greedy_subsets_searcher/kernel/hist_one_byte.cu
    greedy_subsets_searcher/kernel/hist_2_one_byte_5bit.cu
    greedy_subsets_searcher/kernel/hist_2_one_byte_6bit.cu
    greedy_subsets_searcher/kernel/hist_2_one_byte_7bit.cu
    greedy_subsets_searcher/kernel/hist_half_byte.cu
    greedy_subsets_searcher/kernel/hist_binary.cu
    greedy_subsets_searcher/kernel/histogram_utils.cu
    greedy_subsets_searcher/kernel/split_points.cu
    greedy_subsets_searcher/kernel/compute_scores.cu

    add_oblivious_tree_model_feature_parallel.cpp
    histograms_helper.cpp
    helpers.cpp
    pointwise_score_calcer.cpp
    GLOBAL pointwise_kernels.cpp
    GLOBAL pairwise_kernels.cpp
    feature_parallel_pointwise_oblivious_tree.cpp
    oblivious_tree_structure_searcher.cpp
    oblivious_tree_doc_parallel_structure_searcher.cpp
    leaves_estimation/oblivious_tree_leaves_estimator.cpp
    leaves_estimation/step_estimator.cpp
    leaves_estimation/leaves_estimation_helper.cpp
    leaves_estimation/descent_helpers.cpp
    leaves_estimation/doc_parallel_leaves_estimator.cpp
    leaves_estimation/pointwise_oracle.cpp

    boosting_progress_tracker.cpp
    boosting_metric_calcer.cpp
    tree_ctrs.cpp
    ctr_from_tensor_calcer.cpp
    batch_feature_tensor_builder.cpp
    tree_ctrs_dataset.cpp
    tree_ctr_datasets_visitor.cpp
    serialization_helper.cpp
    pointwise_optimization_subsets.cpp

    pairwise_oblivious_trees/pairwise_score_calcer_for_policy.cpp
    pairwise_oblivious_trees/pairwise_scores_calcer.cpp
    pairwise_oblivious_trees/blocked_histogram_helper.cpp
    pairwise_oblivious_trees/pairwise_oblivious_tree.cpp
    pairwise_oblivious_trees/pairwise_optimization_subsets.cpp
    pairwise_oblivious_trees/pairwise_structure_searcher.cpp

)

PEERDIR(
    catboost/cuda/ctrs
    catboost/cuda/cuda_lib
    catboost/cuda/cuda_util
    catboost/cuda/data
    catboost/cuda/gpu_data
    catboost/cuda/models
    catboost/cuda/targets
    catboost/private/libs/ctr_description
    catboost/libs/data
    catboost/libs/helpers
    catboost/private/libs/lapack
    catboost/libs/loggers
    catboost/libs/metrics
    catboost/libs/overfitting_detector
    library/cpp/threading/local_executor
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
