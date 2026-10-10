#include "evaluator.h"

#include <catboost/metal/native/metal_inference.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/model/evaluation_interface.h>
#include <catboost/libs/model/model.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace NCB::NModelEvaluation {
namespace {

// Shared CatBoost preparation supplies numeric, one-hot and CTR buckets. Only
// the tree traversal and summation use the Metal inference runtime. This keeps
// category hashing, NaN modes and full CTR tables identical to native CatBoost.
class TMetalEvaluator final : public IModelEvaluator {
public:
    explicit TMetalEvaluator(const TFullModel& model)
        : Trees(model.ModelTrees)
        , CtrProvider(model.CtrProvider)
        , Oblivious(model.IsOblivious())
    {
        CB_ENSURE(model.GetDimensionsCount() >= 1 && model.GetDimensionsCount() <= 64,
                  "Metal prediction supports model dimensions in [1,64]");
        Dimensions = model.GetDimensionsCount();
        CB_ENSURE(!model.HasTextFeatures() && !model.HasEmbeddingFeatures(),
                  "Metal prediction currently requires numeric/categorical models");
        const auto& data = Trees->GetModelTreeData();
        auto sizes = data->GetTreeSizes();
        CB_ENSURE(sizes.size() <= 1000000, "Metal inference tree count exceeds its limit");
        if (!Oblivious) {
            PrepareNonSymmetricModel();
            return;
        }
        for (int depth : sizes) {
            CB_ENSURE(depth >= 0 && depth <= 16, "Metal prediction supports tree depths through 16");
            SplitStride = std::max(SplitStride, static_cast<ui32>(depth));
            Depths.push_back(depth);
        }
        LeafStride = ui32(1) << SplitStride;
        const ui64 splitCount = ui64(sizes.size()) * SplitStride;
        const ui64 leafCount = ui64(sizes.size()) * LeafStride * Dimensions;
        CB_ENSURE(4 * sizes.size() + 9 * splitCount + 8 * leafCount <= (ui64(1) << 30),
                  "Metal inference model exceeds its 1 GiB memory limit");
        SplitFeatures.resize(splitCount);
        SplitBins.resize(splitCount);
        SplitTypes.resize(splitCount);
        Leaves.resize(leafCount);
        auto repacked = Trees->GetRepackedBins();
        auto starts = data->GetTreeStartOffsets();
        auto leafOffsets = Trees->GetApplyData()->TreeFirstLeafOffsets;
        auto leaves = data->GetLeafValues();
        for (size_t tree = 0; tree < sizes.size(); ++tree) {
            CB_ENSURE(starts[tree] >= 0 && ui64(starts[tree]) + sizes[tree] <= repacked.size(),
                      "Invalid model split offsets");
            for (ui32 level = 0; level < Depths[tree]; ++level) {
                const auto& split = repacked[starts[tree] + level];
                size_t index = tree * SplitStride + level;
                CB_ENSURE(split.SplitIdx > 0 && split.FeatureIndex < Trees->GetEffectiveBinaryFeaturesBucketsCount(),
                          "Invalid repacked Metal model split");
                SplitFeatures[index] = split.FeatureIndex;
                if (split.XorMask) {
                    CB_ENSURE(split.SplitIdx == 255, "Unsupported repacked categorical predicate");
                    SplitTypes[index] = 1;
                    SplitBins[index] = split.XorMask ^ 255;
                } else {
                    SplitBins[index] = split.SplitIdx - 1;
                }
            }
            const ui64 count = (ui64(1) << Depths[tree]) * Dimensions;
            CB_ENSURE(leafOffsets[tree] + count <= leaves.size(), "Invalid model leaf offsets");
            std::copy_n(leaves.data() + leafOffsets[tree], count, Leaves.data() + tree * LeafStride * Dimensions);
        }
    }

    void SetPredictionType(EPredictionType value) override { PredictionType = value; }
    EPredictionType GetPredictionType() const override { return PredictionType; }
    TModelEvaluatorPtr Clone() const override { return new TMetalEvaluator(*this); }
    i32 GetApproxDimension() const override { return Dimensions; }
    size_t GetTreeCount() const override { return Depths.size(); }
    void SetFeatureLayout(const TFeatureLayout& layout) override { FeatureLayout = layout; }
    void SetProperty(TStringBuf name, TStringBuf) override {
        CB_ENSURE(false, "Unknown Metal evaluator property: " << name);
    }

    void Calc(const IQuantizedData* features, size_t start, size_t end, TArrayRef<double> results) const override {
        const auto* quantized = dynamic_cast<const TCPUEvaluatorQuantizedData*>(features);
        CB_ENSURE(quantized, "Metal inference expects shared CatBoost evaluator quantization");
        const size_t rows = quantized->ObjectsCount;
        CB_ENSURE(results.size() == rows * ResultDimension(), "Metal prediction result shape does not match its type");
        CB_ENSURE(start <= end && end <= Depths.size(), "Invalid Metal prediction tree range");
        CB_ENSURE(rows <= (ui64(1) << 27), "Metal prediction row count exceeds its limit");
        if (!rows) return;
        const ui32 featuresCount = Trees->GetEffectiveBinaryFeaturesBucketsCount();
        CB_ENSURE(ui64(rows) * featuresCount <= (ui64(1) << 30), "Metal quantized input exceeds its 1 GiB preparation limit");
        const size_t expectedBlocks = (rows + FORMULA_EVALUATION_BLOCK_SIZE - 1) / FORMULA_EVALUATION_BLOCK_SIZE;
        CB_ENSURE(quantized->BlocksCount == expectedBlocks
                  && (!featuresCount || quantized->BlockStride == featuresCount * FORMULA_EVALUATION_BLOCK_SIZE),
                  "Invalid shared quantized block layout");
        CB_ENSURE(quantized->QuantizedData.GetSize() >= rows * featuresCount,
                  "Shared quantized buffer is too short");
        TVector<ui8> bins(rows * featuresCount);
        auto source = *quantized->QuantizedData;
        for (size_t block = 0; block < expectedBlocks; ++block) {
            const size_t rowOffset = block * FORMULA_EVALUATION_BLOCK_SIZE;
            const size_t count = std::min(FORMULA_EVALUATION_BLOCK_SIZE, rows - rowOffset);
            for (ui32 feature = 0; feature < featuresCount; ++feature) {
                std::memcpy(bins.data() + feature * rows + rowOffset,
                            source.data() + block * quantized->BlockStride + feature * count, count);
            }
        }
        CBMInferenceParams params{static_cast<ui32>(rows), featuresCount, static_cast<ui32>(Depths.size()),
            SplitStride, LeafStride, static_cast<ui32>(start), static_cast<ui32>(end), 65536, 1.0, 0.0};
        CBMInferenceStats stats{};
        char error[2048] = {};
        // Class results have one element per row, while their raw evaluation
        // needs all dimensions. The shared processor supplies that workspace.
        // Scale the raw view before identity postprocessing, including the
        // Class workspace, so vector biases also affect the winning class.
        TEvalResultProcessor processor(rows, results, PredictionType, TScaleAndBias(), Dimensions, rows);
        auto rawResults = processor.GetViewForRawEvaluation(0);
        TVector<double> zeroBias(Dimensions, 0.0);
        const int code = Oblivious
            ? cbm_predict_bins_multidim(&params, Dimensions, zeroBias.data(), zeroBias.size(),
                bins.data(), bins.size(), Depths.data(), Depths.size(),
                SplitFeatures.data(), SplitFeatures.size(), SplitBins.data(), SplitBins.size(),
                SplitTypes.data(), SplitTypes.size(), Leaves.data(), Leaves.size(),
                rawResults.data(), rawResults.size(), &stats, error, sizeof(error))
            : cbm_predict_non_symmetric_bins_multidim(&params, Dimensions, zeroBias.data(), zeroBias.size(),
                bins.data(), bins.size(), Roots.data(), Roots.size(), Nodes.data(), Nodes.size(),
                Leaves.data(), Leaves.size(), rawResults.data(), rawResults.size(), &stats, error, sizeof(error));
        CB_ENSURE(code == 0, "Metal model evaluation failed: " << error);
        ::ApplyScaleAndBias(Trees->GetScaleAndBias(), rawResults, start);
        processor.PostprocessBlock(0, start);
    }

    template <typename FloatAccessor, typename CatAccessor>
    void PrepareAndCalc(FloatAccessor floats, CatAccessor cats, size_t rows,
                        size_t start, size_t end, TArrayRef<double> results, const TFeatureLayout* layout) const {
        const size_t resultDimensions = ResultDimension();
        CB_ENSURE(results.size() == rows * resultDimensions, "Metal prediction result shape does not match its type");
        CB_ENSURE(start <= end && end <= Depths.size(), "Invalid Metal prediction tree range");
        if (!rows) return;
        size_t written = 0;
        ProcessDocsInBlocks(*Trees, CtrProvider, floats, cats, rows, std::min<size_t>(32768, rows),
            [&](size_t count, const TCPUEvaluatorQuantizedData* block) {
                Calc(block, start, end, results.Slice(written * resultDimensions, count * resultDimensions));
                written += count;
            }, layout ? layout : FeatureLayout.Get());
    }

    void CalcFlat(TConstArrayRef<TConstArrayRef<float>> features, size_t start, size_t end,
                  TArrayRef<double> results, const TFeatureLayout* layout) const override {
        auto value = [&](TFeaturePosition position, size_t row) {
            CB_ENSURE(position.FlatIndex >= 0 && size_t(position.FlatIndex) < features[row].size(), "Insufficient flat features");
            return features[row][position.FlatIndex];
        };
        PrepareAndCalc(value, [&](TFeaturePosition p, size_t row) { return ConvertFloatCatFeatureToIntHash(value(p, row)); },
                       features.size(), start, end, results, layout);
    }

    void CalcFlatSingle(TConstArrayRef<float> features, size_t start, size_t end,
                        TArrayRef<double> results, const TFeatureLayout* layout) const override {
        const TConstArrayRef<float> row = features;
        CalcFlat(MakeArrayRef(&row, 1), start, end, results, layout);
    }

    void CalcFlatTransposed(TConstArrayRef<TConstArrayRef<float>> features, size_t start, size_t end,
                            TArrayRef<double> results, const TFeatureLayout* layout) const override {
        CB_ENSURE(results.size() % ResultDimension() == 0, "Invalid Metal transposed result shape");
        auto value = [&](TFeaturePosition position, size_t row) {
            CB_ENSURE(position.FlatIndex >= 0 && size_t(position.FlatIndex) < features.size()
                      && row < features[position.FlatIndex].size(), "Insufficient transposed features");
            return features[position.FlatIndex][row];
        };
        PrepareAndCalc(value, [&](TFeaturePosition p, size_t row) { return ConvertFloatCatFeatureToIntHash(value(p, row)); },
                       results.size() / ResultDimension(), start, end, results, layout);
    }

    template <typename Cat>
    void CalcTyped(TConstArrayRef<TConstArrayRef<float>> floats, TConstArrayRef<TConstArrayRef<Cat>> cats,
                   size_t start, size_t end, TArrayRef<double> results, const TFeatureLayout* layout) const {
        const size_t rows = std::max(floats.size(), cats.size());
        CB_ENSURE((floats.empty() || floats.size() == rows) && (cats.empty() || cats.size() == rows),
                  "Numeric and categorical row counts differ");
        PrepareAndCalc([&](TFeaturePosition p, size_t row) {
            CB_ENSURE(row < floats.size() && p.Index >= 0 && size_t(p.Index) < floats[row].size(), "Insufficient numeric features");
            return floats[row][p.Index];
        }, [&](TFeaturePosition p, size_t row) {
            CB_ENSURE(row < cats.size() && p.Index >= 0 && size_t(p.Index) < cats[row].size(), "Insufficient categorical features");
            if constexpr (std::is_same_v<Cat, int>) return cats[row][p.Index];
            else return static_cast<int>(CalcCatFeatureHash(cats[row][p.Index]));
        }, rows, start, end, results, layout);
    }

    void Calc(TConstArrayRef<TConstArrayRef<float>> floats, TConstArrayRef<TConstArrayRef<int>> cats,
              size_t start, size_t end, TArrayRef<double> results, const TFeatureLayout* layout) const override {
        CalcTyped(floats, cats, start, end, results, layout);
    }
    void Calc(TConstArrayRef<TConstArrayRef<float>> floats, TConstArrayRef<TConstArrayRef<TStringBuf>> cats,
              size_t start, size_t end, TArrayRef<double> results, const TFeatureLayout* layout) const override {
        CalcTyped(floats, cats, start, end, results, layout);
    }
    void Calc(TConstArrayRef<TConstArrayRef<float>> floats, TConstArrayRef<TConstArrayRef<TStringBuf>> cats,
              TConstArrayRef<TConstArrayRef<TStringBuf>>, size_t start, size_t end,
              TArrayRef<double> results, const TFeatureLayout* layout) const override {
        CalcTyped(floats, cats, start, end, results, layout);
    }
    void Calc(TConstArrayRef<TConstArrayRef<float>> floats, TConstArrayRef<TConstArrayRef<TStringBuf>> cats,
              TConstArrayRef<TConstArrayRef<TStringBuf>>, TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>>,
              size_t start, size_t end, TArrayRef<double> results, const TFeatureLayout* layout) const override {
        CalcTyped(floats, cats, start, end, results, layout);
    }
    void CalcWithHashedCatAndTextAndEmbeddings(TConstArrayRef<TConstArrayRef<float>> floats,
              TConstArrayRef<TConstArrayRef<int>> cats, TConstArrayRef<TConstArrayRef<TStringBuf>>,
              TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>>, size_t start, size_t end,
              TArrayRef<double> results, const TFeatureLayout* layout) const override {
        CalcTyped(floats, cats, start, end, results, layout);
    }

    void Quantize(TConstArrayRef<TConstArrayRef<float>> features, IQuantizedData* destination) const override {
        auto* output = dynamic_cast<TCPUEvaluatorQuantizedData*>(destination);
        CB_ENSURE(output, "Metal quantization expects shared CatBoost evaluator buffers");
        const size_t rows = features.size();
        const size_t buckets = Trees->GetEffectiveBinaryFeaturesBucketsCount();
        CB_ENSURE(rows <= (ui64(1) << 27) && ui64(rows) * buckets <= (ui64(1) << 30), "Metal quantization exceeds memory limit");
        output->ObjectsCount = rows;
        output->BlocksCount = (rows + FORMULA_EVALUATION_BLOCK_SIZE - 1) / FORMULA_EVALUATION_BLOCK_SIZE;
        output->BlockStride = buckets * FORMULA_EVALUATION_BLOCK_SIZE;
        output->QuantizedData = TMaybeOwningArrayHolder<ui8>::CreateOwning(TVector<ui8>(rows * buckets));
        if (!rows) return;
        size_t written = 0;
        auto value = [&](TFeaturePosition p, size_t row) {
            CB_ENSURE(p.FlatIndex >= 0 && size_t(p.FlatIndex) < features[row].size(), "Insufficient flat features");
            return features[row][p.FlatIndex];
        };
        ProcessDocsInBlocks(*Trees, CtrProvider, value,
            [&](TFeaturePosition p, size_t row) { return ConvertFloatCatFeatureToIntHash(value(p, row)); },
            rows, std::min<size_t>(32768, rows), [&](size_t count, const TCPUEvaluatorQuantizedData* block) {
                std::copy_n((*block->QuantizedData).data(), count * buckets, (*output->QuantizedData).data() + written * buckets);
                written += count;
            }, FeatureLayout.Get());
    }

    void CalcLeafIndexesSingle(TConstArrayRef<float>, TConstArrayRef<TStringBuf>, size_t, size_t,
                              TArrayRef<TCalcerIndexType>, const TFeatureLayout*) const override {
        CB_ENSURE(false, "Metal leaf-index output is not implemented yet");
    }
    void CalcLeafIndexes(TConstArrayRef<TConstArrayRef<float>>, TConstArrayRef<TConstArrayRef<TStringBuf>>,
                         size_t, size_t, TArrayRef<TCalcerIndexType>, const TFeatureLayout*) const override {
        CB_ENSURE(false, "Metal leaf-index output is not implemented yet");
    }
    void CalcLeafIndexes(const IQuantizedData*, size_t, size_t, TArrayRef<TCalcerIndexType>) const override {
        CB_ENSURE(false, "Metal leaf-index output is not implemented yet");
    }

private:
    void PrepareNonSymmetricModel() {
        const auto& data = Trees->GetModelTreeData();
        const auto steps = data->GetNonSymmetricStepNodes();
        const auto starts = data->GetTreeStartOffsets();
        const auto sizes = data->GetTreeSizes();
        const auto nodeLeaves = data->GetNonSymmetricNodeIdToLeafId();
        const auto repacked = Trees->GetRepackedBins();
        const auto values = data->GetLeafValues();
        CB_ENSURE(starts.size() == sizes.size() && nodeLeaves.size() == steps.size() &&
            repacked.size() == steps.size(), "Invalid native non-symmetric model arrays");
        CB_ENSURE(ui64(steps.size()) * 2 * sizeof(CBMInferenceNode) + values.size() * 8 + sizes.size() * 4 <= (ui64(1) << 30),
            "Metal variable-tree model exceeds its 1 GiB memory limit");
        Nodes.resize(steps.size());
        Roots.reserve(sizes.size());
        Depths.resize(sizes.size(), 0);
        LeafStride = 0;
        Leaves.assign(values.begin(), values.end());
        auto leafNode = [&](ui32 index) {
            CB_ENSURE(index < nodeLeaves.size() && nodeLeaves[index] % Dimensions == 0 &&
                ui64(nodeLeaves[index]) + Dimensions <= values.size(), "Invalid native variable-tree leaf offset");
            return CBMInferenceNode{0, 0, 0, 0, 0, nodeLeaves[index] / Dimensions};
        };
        for (size_t tree = 0; tree < sizes.size(); ++tree) {
            CB_ENSURE(starts[tree] >= 0 && sizes[tree] > 0 && ui64(starts[tree]) + sizes[tree] <= steps.size(),
                "Invalid native variable-tree start or size");
            const ui32 begin = starts[tree], end = begin + sizes[tree];
            Roots.push_back(begin);
            for (ui32 index = begin; index < end; ++index) {
                const auto& step = steps[index];
                if (!step.LeftSubtreeDiff && !step.RightSubtreeDiff) {
                    Nodes[index] = leafNode(index);
                    continue;
                }
                const auto& split = repacked[index];
                CB_ENSURE(split.SplitIdx > 0 && split.FeatureIndex < Trees->GetEffectiveBinaryFeaturesBucketsCount(),
                    "Invalid repacked variable-tree split");
                CBMInferenceNode node{split.FeatureIndex, ui32(split.SplitIdx - 1), 0, 0, 0, Max<ui32>()};
                if (split.XorMask) {
                    CB_ENSURE(split.SplitIdx == 255, "Unsupported repacked variable-tree categorical predicate");
                    node.type = 1; node.bin = split.XorMask ^ 255;
                }
                auto child = [&](ui32 difference) {
                    if (difference) {
                        CB_ENSURE(ui64(index) + difference < end, "Native variable-tree child leaves its tree");
                        return index + difference;
                    }
                    // Native compact nodes fold one terminal child into its
                    // parent. Expand only that leaf, retaining linear storage.
                    const ui32 result = Nodes.size();
                    Nodes.push_back(leafNode(index));
                    return result;
                };
                node.left = child(step.LeftSubtreeDiff);
                node.right = child(step.RightSubtreeDiff);
                Nodes[index] = node;
            }
        }
    }

    ui32 ResultDimension() const {
        return PredictionType == EPredictionType::Class ? 1 : Dimensions;
    }
    TCOWTreeWrapper Trees;
    TIntrusivePtr<ICtrProvider> CtrProvider;
    bool Oblivious = true;
    TMaybe<TFeatureLayout> FeatureLayout;
    EPredictionType PredictionType = EPredictionType::RawFormulaVal;
    ui32 Dimensions = 1, SplitStride = 0, LeafStride = 1;
    TVector<ui32> Depths, SplitFeatures, SplitBins;
    TVector<ui8> SplitTypes;
    TVector<double> Leaves;
    TVector<ui32> Roots;
    TVector<CBMInferenceNode> Nodes;
};

TEvaluationBackendFactory::TRegistrator<TMetalEvaluator> MetalEvaluationBackendRegistrator(EFormulaEvaluatorType::GPU);
} // namespace
void* MetalEvaluationBackendRegistratorPointer = &MetalEvaluationBackendRegistrator;
} // namespace NCB::NModelEvaluation
