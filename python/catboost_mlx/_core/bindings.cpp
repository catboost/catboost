// python/catboost_mlx/_core/bindings.cpp — Nanobind Python bindings.
//
// Exposes TTrainConfig, TTrainResultAPI, and train() to Python.
// The train() function accepts numpy arrays and calls TrainFromArrays().
// GIL is released during Metal GPU training.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/ndarray.h>

#include "catboost/mlx/train_api.h"

namespace nb = nanobind;

// ndarray type aliases for zero-copy numpy → C++ pointer access.
using FloatArray2D = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using FloatArray1D = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using UInt32Array1D = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

NB_MODULE(_core, m) {
    m.doc() = "CatBoost-MLX in-process training and prediction bindings.";

    // ── TTrainConfig binding ────────────────────────────────────────────────
    nb::class_<TTrainConfig>(m, "TrainConfig")
        .def(nb::init<>())
        .def_rw("num_iterations",        &TTrainConfig::NumIterations)
        .def_rw("max_depth",             &TTrainConfig::MaxDepth)
        .def_rw("learning_rate",         &TTrainConfig::LearningRate)
        .def_rw("l2_reg_lambda",         &TTrainConfig::L2RegLambda)
        .def_rw("max_bins",              &TTrainConfig::MaxBins)
        .def_rw("loss_type",             &TTrainConfig::LossType)
        .def_rw("group_col",             &TTrainConfig::GroupCol)
        .def_rw("weight_col",            &TTrainConfig::WeightCol)
        .def_rw("eval_fraction",         &TTrainConfig::EvalFraction)
        .def_rw("early_stopping_patience", &TTrainConfig::EarlyStoppingPatience)
        .def_rw("subsample_ratio",       &TTrainConfig::SubsampleRatio)
        .def_rw("colsample_by_tree",     &TTrainConfig::ColsampleByTree)
        .def_rw("random_seed",           &TTrainConfig::RandomSeed)
        .def_rw("random_strength",       &TTrainConfig::RandomStrength)
        .def_rw("bootstrap_type",        &TTrainConfig::BootstrapType)
        .def_rw("bagging_temperature",   &TTrainConfig::BaggingTemperature)
        .def_rw("mvs_reg",              &TTrainConfig::MvsReg)
        .def_rw("nan_mode",              &TTrainConfig::NanMode)
        .def_rw("use_ctr",               &TTrainConfig::UseCtr)
        .def_rw("ctr_prior",             &TTrainConfig::CtrPrior)
        .def_rw("max_onehot_size",       &TTrainConfig::MaxOneHotSize)
        .def_rw("min_data_in_leaf",      &TTrainConfig::MinDataInLeaf)
        .def_rw("monotone_constraints",  &TTrainConfig::MonotoneConstraints)
        .def_rw("grow_policy",           &TTrainConfig::GrowPolicy)
        .def_rw("max_leaves",            &TTrainConfig::MaxLeaves)
        .def_rw("score_function",        &TTrainConfig::ScoreFunction)
        .def_rw("snapshot_path",         &TTrainConfig::SnapshotPath)
        .def_rw("snapshot_interval",     &TTrainConfig::SnapshotInterval)
        .def_rw("verbose",               &TTrainConfig::Verbose)
        .def_rw("compute_feature_importance", &TTrainConfig::ComputeFeatureImportance);

    // ── TTrainResultAPI binding ─────────────────────────────────────────────
    nb::class_<TTrainResultAPI>(m, "TrainResult")
        .def_ro("final_train_loss",      &TTrainResultAPI::FinalTrainLoss)
        .def_ro("final_test_loss",       &TTrainResultAPI::FinalTestLoss)
        .def_ro("best_iteration",        &TTrainResultAPI::BestIteration)
        .def_ro("trees_built",           &TTrainResultAPI::TreesBuilt)
        .def_ro("model_json",            &TTrainResultAPI::ModelJSON)
        .def_ro("feature_importance",    &TTrainResultAPI::FeatureImportance)
        .def_ro("train_loss_history",    &TTrainResultAPI::TrainLossHistory)
        .def_ro("eval_loss_history",     &TTrainResultAPI::EvalLossHistory)
        .def_ro("feature_names",         &TTrainResultAPI::FeatureNames)
        .def_ro("grad_ms",              &TTrainResultAPI::GradMs)
        .def_ro("tree_search_ms",        &TTrainResultAPI::TreeSearchMs)
        .def_ro("leaf_ms",              &TTrainResultAPI::LeafMs)
        .def_ro("apply_ms",             &TTrainResultAPI::ApplyMs);

    // ── train() binding ─────────────────────────────────────────────────────
    m.def(
        "train",
        [](FloatArray2D features,
           FloatArray1D targets,
           const std::vector<std::string>& feature_names,
           const std::vector<bool>& is_categorical,
           nb::object weights_obj,
           nb::object group_ids_obj,
           const std::vector<std::unordered_map<std::string, uint32_t>>& cat_hash_maps,
           nb::object val_features_obj,
           nb::object val_targets_obj,
           const TTrainConfig& config)
        -> TTrainResultAPI
        {
            uint32_t n_docs = static_cast<uint32_t>(features.shape(0));
            uint32_t n_features = static_cast<uint32_t>(features.shape(1));

            const float* feat_ptr = features.data();
            const float* tgt_ptr  = targets.data();

            // Optional weights
            std::vector<float> weights_vec;
            if (!weights_obj.is_none()) {
                auto w = nb::cast<FloatArray1D>(weights_obj);
                weights_vec.assign(w.data(), w.data() + w.size());
            }

            // Optional group IDs
            std::vector<uint32_t> group_ids_vec;
            if (!group_ids_obj.is_none()) {
                auto g = nb::cast<UInt32Array1D>(group_ids_obj);
                group_ids_vec.assign(g.data(), g.data() + g.size());
            }

            // Optional validation set
            const float* val_feat_ptr = nullptr;
            const float* val_tgt_ptr  = nullptr;
            uint32_t val_docs = 0;
            if (!val_features_obj.is_none() && !val_targets_obj.is_none()) {
                auto vf = nb::cast<FloatArray2D>(val_features_obj);
                auto vt = nb::cast<FloatArray1D>(val_targets_obj);
                val_feat_ptr = vf.data();
                val_tgt_ptr  = vt.data();
                val_docs = static_cast<uint32_t>(vt.size());
            }

            TTrainResultAPI result;
            // Release the GIL while Metal GPU training runs.
            // All Metal/MLX calls are thread-safe; Python objects are not
            // accessed during training.
            {
                nb::gil_scoped_release release;
                result = TrainFromArrays(
                    feat_ptr, tgt_ptr,
                    feature_names, is_categorical,
                    weights_vec, group_ids_vec, cat_hash_maps,
                    n_docs, n_features,
                    val_feat_ptr, val_tgt_ptr, val_docs,
                    config
                );
            }
            return result;
        },
        nb::arg("features"),
        nb::arg("targets"),
        nb::arg("feature_names"),
        nb::arg("is_categorical"),
        nb::arg("weights")       = nb::none(),
        nb::arg("group_ids")     = nb::none(),
        nb::arg("cat_hash_maps") = std::vector<std::unordered_map<std::string, uint32_t>>{},
        nb::arg("val_features")  = nb::none(),
        nb::arg("val_targets")   = nb::none(),
        nb::arg("config")        = TTrainConfig{},
        "Train a GBDT model on Apple Silicon GPU.\n\n"
        "Accepts numpy arrays directly (zero-copy for contiguous float32).\n"
        "Releases the GIL during Metal GPU training."
    );
}
