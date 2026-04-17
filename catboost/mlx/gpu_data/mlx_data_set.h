#pragma once

// GPU-resident dataset for MLX Metal backend.
// Holds all data needed for GBDT training on Apple Silicon GPU.

#include <catboost/mlx/gpu_data/compressed_index.h>
#include <catboost/mlx/gpu_data/gpu_structures.h>
#include <catboost/mlx/gpu_data/mlx_device.h>

#include <util/generic/vector.h>

#include <vector>

namespace NCatboostMlx {

    class TMLXDataSet {
    public:
        TMLXDataSet() = default;

        // --- Feature data ---

        void SetCompressedIndex(TMLXCompressedIndex&& index) {
            CompressedIndex_ = std::move(index);
        }

        const TMLXCompressedIndex& GetCompressedIndex() const {
            return CompressedIndex_;
        }

        // Return the raw compressed feature buffer (mx::array) for kernel access
        const mx::array& GetCompressedData() const { return CompressedIndex_.GetCompressedData(); }

        // --- Target and weight data ---

        void SetTargets(const float* targets, ui32 numDocs) {
            Targets_ = mx::array(targets, {static_cast<int>(numDocs)}, mx::float32);
            TMLXDevice::EvalAtBoundary(Targets_);
        }

        void SetWeights(const float* weights, ui32 numDocs) {
            Weights_ = mx::array(weights, {static_cast<int>(numDocs)}, mx::float32);
            HasWeights_ = true;
            TMLXDevice::EvalAtBoundary(Weights_);
        }

        void SetUniformWeights(ui32 numDocs) {
            Weights_ = mx::ones({static_cast<int>(numDocs)}, mx::float32);
            HasWeights_ = false;
            TMLXDevice::EvalAtBoundary(Weights_);
        }

        const mx::array& GetTargets() const { return Targets_; }
        ui32 GetNumDocs() const { return CompressedIndex_.GetNumDocs(); }
        ui32 GetNumUi32PerDoc() const { return CompressedIndex_.GetNumUi32PerDoc(); }
        const mx::array& GetWeights() const { return Weights_; }
        bool HasWeights() const { return HasWeights_; }

        // --- Group data (for ranking losses: PairLogit, YetiRank) ---

        /// Set query/group offsets for ranking losses.
        /// groupOffsets has size numGroups + 1; groupOffsets[g] is the first doc index
        /// of group g, groupOffsets[numGroups] == numDocs (sentinel).
        void SetGroupData(const std::vector<ui32>& groupOffsets, ui32 numGroups) {
            GroupOffsets_ = groupOffsets;
            NumGroups_ = numGroups;
        }

        const std::vector<ui32>& GetGroupOffsets() const { return GroupOffsets_; }
        ui32 GetNumGroups() const { return NumGroups_; }

        /// Returns true when group data has been populated (required for ranking losses).
        bool HasGroups() const { return NumGroups_ > 0; }

        // --- Training state (mutable cursor) ---

        // Initialize prediction cursor to zeros
        void InitCursor(ui32 numDocs, ui32 approxDimension) {
            Cursor_ = mx::zeros(
                {static_cast<int>(approxDimension), static_cast<int>(numDocs)},
                mx::float32
            );
            TMLXDevice::EvalAtBoundary(Cursor_);
        }

        mx::array& GetCursor() { return Cursor_; }
        const mx::array& GetCursor() const { return Cursor_; }

        // --- Gradient/Hessian buffers ---

        void InitGradients(ui32 numDocs, ui32 approxDimension) {
            Gradients_ = mx::zeros(
                {static_cast<int>(approxDimension), static_cast<int>(numDocs)},
                mx::float32
            );
            Hessians_ = mx::zeros(
                {static_cast<int>(approxDimension), static_cast<int>(numDocs)},
                mx::float32
            );
            TMLXDevice::EvalAtBoundary({Gradients_, Hessians_});
        }

        mx::array& GetGradients() { return Gradients_; }
        mx::array& GetHessians() { return Hessians_; }
        const mx::array& GetGradients() const { return Gradients_; }
        const mx::array& GetHessians() const { return Hessians_; }

        // --- Document partitioning (leaf assignments) ---

        // Initialize all docs to leaf 0
        void InitPartitions(ui32 numDocs) {
            Partitions_ = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
            TMLXDevice::EvalAtBoundary(Partitions_);
        }

        mx::array& GetPartitions() { return Partitions_; }
        const mx::array& GetPartitions() const { return Partitions_; }

    private:
        TMLXCompressedIndex CompressedIndex_;

        mx::array Targets_;       // [numDocs] float32
        mx::array Weights_;       // [numDocs] float32
        mx::array Cursor_;        // [approxDim, numDocs] float32 — current model predictions
        mx::array Gradients_;     // [approxDim, numDocs] float32
        mx::array Hessians_;      // [approxDim, numDocs] float32
        mx::array Partitions_;    // [numDocs] uint32 — leaf index per document

        bool HasWeights_ = false;

        std::vector<ui32> GroupOffsets_;  // size = NumGroups_ + 1
        ui32 NumGroups_ = 0;
    };

}  // namespace NCatboostMlx
