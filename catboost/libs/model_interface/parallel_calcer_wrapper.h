#pragma once
/**
 * ParallelModelCalcerWrapper
 *
 * Header-only C++ wrapper around the CatBoost C API that adds multi-threaded
 * batch prediction. The class manages its own ModelCalcerHandle (same pattern
 * as ModelCalcerWrapper) and exposes parallel variants of every batch-predict
 * method, each accepting a `numThreads` parameter.
 *
 * Thread-safety
 * -------------
 * The CatBoost evaluator is stateless per call: multiple threads may call
 * CalcModelPrediction* on the same handle simultaneously. Each parallel worker
 * writes into a disjoint slice of the pre-allocated result vector, so no
 * mutex is needed.
 *
 * numThreads semantics
 * --------------------
 *   0  -> use std::thread::hardware_concurrency()
 *   1  -> single-threaded fast path (no thread overhead)
 *   N  -> exactly N worker threads (clamped to docCount)
 */

#include "c_api.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

class ParallelModelCalcerWrapper {
public:
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /** Create an empty (unloaded) calcer. */
    ParallelModelCalcerWrapper()
        : Handle_(CalcerHolderType(ModelCalcerCreate(), ModelCalcerDelete))
    {}

    /**
     * Load model from file.
     * @param filename  Path to the CatBoost model binary.
     */
    explicit ParallelModelCalcerWrapper(const std::string& filename)
        : Handle_(CalcerHolderType(ModelCalcerCreate(), ModelCalcerDelete))
    {
        if (!LoadFullModelFromFile(Handle_.get(), filename.c_str())) {
            throw std::runtime_error(GetErrorString());
        }
        InitProps();
    }

    /**
     * Load model from a memory buffer.
     * @param binaryBuffer      Pointer to the buffer.
     * @param binaryBufferSize  Buffer size in bytes.
     */
    explicit ParallelModelCalcerWrapper(const void* binaryBuffer,
                                        size_t binaryBufferSize)
        : Handle_(CalcerHolderType(ModelCalcerCreate(), ModelCalcerDelete))
    {
        if (!LoadFullModelFromBuffer(Handle_.get(), binaryBuffer,
                                     binaryBufferSize)) {
            throw std::runtime_error(GetErrorString());
        }
        InitProps();
    }

    // -----------------------------------------------------------------------
    // Init helpers
    // -----------------------------------------------------------------------

    bool InitFromFile(const std::string& filename) {
        if (!LoadFullModelFromFile(Handle_.get(), filename.c_str())) {
            return false;
        }
        InitProps();
        return true;
    }

    bool InitFromMemory(const void* pointer, size_t size) {
        if (!LoadFullModelFromBuffer(Handle_.get(), pointer, size)) {
            return false;
        }
        InitProps();
        return true;
    }

    // -----------------------------------------------------------------------
    // Model metadata
    // -----------------------------------------------------------------------

    size_t GetTreeCount()              const { return ::GetTreeCount(Handle_.get()); }
    size_t GetFloatFeaturesCount()     const { return ::GetFloatFeaturesCount(Handle_.get()); }
    size_t GetCatFeaturesCount()       const { return ::GetCatFeaturesCount(Handle_.get()); }
    size_t GetTextFeaturesCount()      const { return ::GetTextFeaturesCount(Handle_.get()); }
    size_t GetEmbeddingFeaturesCount() const { return ::GetEmbeddingFeaturesCount(Handle_.get()); }
    size_t GetDimensionsCount()        const { return DimensionsCount_; }

    // -----------------------------------------------------------------------
    // Single-object helpers (unchanged semantics, delegate to C API)
    // -----------------------------------------------------------------------

    /**
     * Predict on a single flat feature vector.
     * Flat = float and categorical features in one array.
     * Only works for single-output models.
     */
    double CalcFlat(const std::vector<float>& features) const {
        double result = 0.0;
        const float* ptr = features.data();
        if (!CalcModelPredictionFlat(Handle_.get(), 1, &ptr, features.size(),
                                     &result, 1)) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    /**
     * Predict on a single flat feature vector (multi-output).
     */
    std::vector<double> CalcFlatMulti(const std::vector<float>& features) const {
        std::vector<double> result(DimensionsCount_, 0.0);
        const float* ptr = features.data();
        if (!CalcModelPredictionFlat(Handle_.get(), 1, &ptr, features.size(),
                                     result.data(), DimensionsCount_)) {
            throw std::runtime_error(GetErrorString());
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Batch helpers — single-threaded (mirror ModelCalcerWrapper)
    // -----------------------------------------------------------------------

    /**
     * Predict on a batch of flat feature vectors (single-threaded).
     * @param features  One row per object; all rows must have the same length.
     * @return          Flat result vector of size features.size() * DimensionsCount.
     */
    std::vector<double> CalcFlat(
        const std::vector<std::vector<float>>& features) const
    {
        return CalcFlatParallel(features, /*numThreads=*/1);
    }

    /**
     * Predict on a batch of (float, categorical) feature vectors
     * (single-threaded).
     */
    std::vector<double> Calc(
        const std::vector<std::vector<float>>& floatFeatures,
        const std::vector<std::vector<std::string>>& catFeatures = {}) const
    {
        return CalcParallel(floatFeatures, catFeatures, /*numThreads=*/1);
    }

    // -----------------------------------------------------------------------
    // Parallel batch prediction — flat features
    // -----------------------------------------------------------------------

    /**
     * Predict on a batch of flat feature vectors using multiple threads.
     *
     * The batch is split into `numThreads` contiguous chunks; each chunk is
     * evaluated on a separate std::thread writing into a disjoint slice of
     * the result vector (no mutex needed).
     *
     * @param features    One row per object; all rows must have the same length.
     * @param numThreads  0 = hardware_concurrency, 1 = no threads spawned.
     * @return            Flat result vector of size features.size() * DimensionsCount.
     */
    std::vector<double> CalcFlatParallel(
        const std::vector<std::vector<float>>& features,
        size_t numThreads = 0) const
    {
        const size_t docCount = features.size();
        if (docCount == 0) return {};

        const size_t dims = DimensionsCount_;
        std::vector<double> result(docCount * dims, 0.0);
        numThreads = ResolveThreadCount(numThreads, docCount);

        RunInParallel(numThreads, docCount, [&](size_t begin, size_t end) {
            std::vector<const float*> ptrs;
            ptrs.reserve(end - begin);
            size_t flatVecSize = 0;
            for (size_t i = begin; i < end; ++i) {
                flatVecSize = features[i].size();
                ptrs.push_back(features[i].data());
            }
            if (!CalcModelPredictionFlat(
                    Handle_.get(),
                    end - begin,
                    ptrs.data(), flatVecSize,
                    result.data() + begin * dims,
                    (end - begin) * dims)) {
                throw std::runtime_error(GetErrorString());
            }
        });

        return result;
    }

    // -----------------------------------------------------------------------
    // Parallel batch prediction — float + categorical features
    // -----------------------------------------------------------------------

    /**
     * Predict on a batch of (float, categorical) feature vectors using
     * multiple threads.
     *
     * @param floatFeatures  One row per object.
     * @param catFeatures    One row per object (may be empty).
     * @param numThreads     0 = hardware_concurrency, 1 = no threads spawned.
     * @return               Flat result vector.
     */
    std::vector<double> CalcParallel(
        const std::vector<std::vector<float>>& floatFeatures,
        const std::vector<std::vector<std::string>>& catFeatures = {},
        size_t numThreads = 0) const
    {
        const size_t docCount = floatFeatures.size();
        if (docCount == 0) return {};

        const size_t dims = DimensionsCount_;
        std::vector<double> result(docCount * dims, 0.0);
        numThreads = ResolveThreadCount(numThreads, docCount);

        RunInParallel(numThreads, docCount, [&](size_t begin, size_t end) {
            // Float pointers
            std::vector<const float*> floatPtrs;
            floatPtrs.reserve(end - begin);
            size_t floatFeatureCount = 0;
            for (size_t i = begin; i < end; ++i) {
                floatFeatureCount = floatFeatures[i].size();
                floatPtrs.push_back(floatFeatures[i].data());
            }

            // Cat pointers (flat layout required by the C API)
            size_t catFeatureCount = 0;
            std::vector<const char*> catPtrsFlat;
            std::vector<const char**> catPtrPtrs;
            if (!catFeatures.empty()) {
                catFeatureCount = catFeatures[begin].size();
                catPtrsFlat.reserve((end - begin) * catFeatureCount);
                catPtrPtrs.reserve(end - begin);
                for (size_t i = begin; i < end; ++i) {
                    catPtrPtrs.push_back(
                        catPtrsFlat.data() + catPtrsFlat.size());
                    for (const auto& s : catFeatures[i]) {
                        catPtrsFlat.push_back(s.data());
                    }
                }
            }

            if (!CalcModelPrediction(
                    Handle_.get(),
                    end - begin,
                    floatPtrs.data(), floatFeatureCount,
                    catFeatures.empty() ? nullptr : catPtrPtrs.data(),
                    catFeatureCount,
                    result.data() + begin * dims,
                    (end - begin) * dims)) {
                throw std::runtime_error(GetErrorString());
            }
        });

        return result;
    }

private:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    void InitProps() {
        DimensionsCount_ = ::GetDimensionsCount(Handle_.get());
    }

    /** Resolve 0 -> hardware_concurrency; clamp to docCount. */
    static size_t ResolveThreadCount(size_t requested, size_t docCount) {
        if (requested == 0) {
            unsigned hw = std::thread::hardware_concurrency();
            requested = (hw > 0) ? static_cast<size_t>(hw) : 1u;
        }
        return std::min(requested, docCount);
    }

    /**
     * Partition [0, docCount) into `numThreads` contiguous chunks and run
     * worker(begin, end) on each chunk in a separate std::thread.
     * Exceptions thrown by workers are re-thrown in the calling thread.
     */
    template <typename Worker>
    static void RunInParallel(size_t numThreads, size_t docCount,
                               Worker&& worker)
    {
        if (numThreads == 1) {
            worker(0, docCount);
            return;
        }

        std::vector<std::thread> threads;
        threads.reserve(numThreads);
        std::vector<std::exception_ptr> errors(numThreads, nullptr);

        const size_t chunkSize = (docCount + numThreads - 1) / numThreads;

        for (size_t t = 0; t < numThreads; ++t) {
            const size_t begin = t * chunkSize;
            const size_t end   = std::min(begin + chunkSize, docCount);
            if (begin >= end) break;

            threads.emplace_back(
                [&worker, &errors, t, begin, end]() {
                    try {
                        worker(begin, end);
                    } catch (...) {
                        errors[t] = std::current_exception();
                    }
                });
        }

        for (auto& th : threads) th.join();

        for (const auto& ep : errors) {
            if (ep) std::rethrow_exception(ep);
        }
    }

    // -----------------------------------------------------------------------
    // Data members
    // -----------------------------------------------------------------------

    using CalcerHolderType =
        std::unique_ptr<ModelCalcerHandle,
                        std::function<void(ModelCalcerHandle*)>>;

    CalcerHolderType Handle_;
    size_t DimensionsCount_ = 0;
};

