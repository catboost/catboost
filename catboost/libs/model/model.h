#pragma once

#include "fwd.h"
#include "ctr_provider.h"
#include "evaluation_interface.h"
#include "features.h"
#include "online_ctr.h"
#include "scale_and_bias.h"
#include "split.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/model/enums.h>

#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/text_features/text_processing_collection.h>
#include <catboost/private/libs/embedding_features/embedding_processing_collection.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/stream/fwd.h>
#include <util/stream/mem.h>
#include <util/system/spinlock.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <tuple>


class TModelPartsCachingSerializer;

/*!
    \brief Oblivious tree model structure

    This structure contains the data about tree conditions and leaf values.
    We use oblivious trees - symmetric trees that has the same binary condition on each level.
    So each leaf index is determined by binary vector with length equal to evaluated tree depth.

    That allows us to evaluate model predictions very fast (even without planned SIMD optimizations)
    compared to asymmetric trees.

    Our oblivious tree model can contain float, one-hot and CTR binary conditions:
    - Float condition - float feature value is greater than float border
    - One-hot condition - hashed cat feature value is equal to some value
    - CTR condition - calculated ctr is greater than float border
    You can read about CTR calculation in ctr_provider.h

    FloatFeatures, OneHotFeatures and CtrFeatures form binary features(or binary conditions) sequence.
    Information about tree structure is stored in 3 integer vectors:
    TreeSplits, TreeSizes, TreeStartOffsets.
    - TreeSplits - holds all binary feature indexes from all the trees.
    - TreeSizes - holds tree depth.
    - TreeStartOffsets - holds offset of first tree split in TreeSplits vector
*/
struct TRepackedBin {
    ui16 FeatureIndex = 0;
    ui8 XorMask = 0;
    ui8 SplitIdx = 0;

    TRepackedBin& operator=(const NCatBoostFbs::TRepackedBin*);
};

constexpr ui32 MAX_VALUES_PER_BIN = 254;

constexpr double DEFAULT_BINCLASS_PROBABILITY_THRESHOLD = 0.5;
constexpr double DEFAULT_BINCLASS_LOGIT_THRESHOLD = 0;

// If selected diff is 0 we are in the last node in path
struct TNonSymmetricTreeStepNode {
    static constexpr ui16 InvalidDiff = Max<ui16>();
    static constexpr ui16 TerminalMarker = 0;

    ui16 LeftSubtreeDiff = InvalidDiff;
    ui16 RightSubtreeDiff = InvalidDiff;

    static TNonSymmetricTreeStepNode TerminalNodeSteps() {
        return TNonSymmetricTreeStepNode{0, 0};
    }

    TNonSymmetricTreeStepNode& operator=(const NCatBoostFbs::TNonSymmetricTreeStepNode* stepNode);

    bool operator==(const TNonSymmetricTreeStepNode& other) const {
        return std::tie(LeftSubtreeDiff, RightSubtreeDiff)
            == std::tie(other.LeftSubtreeDiff, other.RightSubtreeDiff);
    }
};

struct IModelTreeData {
    enum class ECloningPolicy { Default, CloneAsSolid, CloneAsOpaque };

    //! Split values
    virtual TConstArrayRef<int> GetTreeSplits() const = 0;

    //! Tree sizes
    virtual TConstArrayRef<int> GetTreeSizes() const = 0;

    //! Offset of first split in TreeSplits array
    virtual TConstArrayRef<int> GetTreeStartOffsets() const = 0;

    //! Steps in a non-symmetric tree.
    //! If at least one diff in a step node is zero, it's a terminal node and has a value.
    //! If both diffs are zero, the corresponding split condition (in the RepackedBins vector) may be invalid.
    virtual TConstArrayRef<TNonSymmetricTreeStepNode> GetNonSymmetricStepNodes() const = 0;

    //! Holds a value index (in the LeafValues vector) for each terminal node in a non-symmetric tree.
    //! For multiclass models holds indexes for 0-class.
    virtual TConstArrayRef<ui32> GetNonSymmetricNodeIdToLeafId() const = 0;

    //! Leaf values layout: [treeIndex][leafId * ApproxDimension + dimension]
    virtual TConstArrayRef<double> GetLeafValues() const = 0;

    /**
     * Leaf Weights are sums of weights or group weights of samples from the learn dataset that go to that leaf.
     * This information can be absent (this vector will be empty) in some models:
     *   - Loaded from CoreML format
     *   - Old models trained on GPU (trained with catboost version < 0.9.x)
     *
     *  layout: [treeIndex][leafId]
     */
    virtual TConstArrayRef<double> GetLeafWeights() const = 0;

    virtual void SetTreeSplits(const TVector<int>&) = 0;
    virtual void SetTreeSizes(const TVector<int>&) = 0;
    virtual void SetTreeStartOffsets(const TVector<int>&) = 0;
    virtual void SetNonSymmetricStepNodes(const TVector<TNonSymmetricTreeStepNode>&) = 0;
    virtual void SetNonSymmetricNodeIdToLeafId(const TVector<ui32>&) = 0;
    virtual void SetLeafValues(const TVector<double>&) = 0;
    virtual void SetLeafWeights(const TVector<double>&) = 0;

    virtual THolder<IModelTreeData> Clone(ECloningPolicy) const = 0;
    virtual ~IModelTreeData() = default;
};

struct TModelTrees {
public:
    /**
     * This structure stores model runtime data. Should be kept up to date
     */
    struct TRuntimeData {
        /**
         * List of all binary with indexes corresponding to TreeSplits values
         */
        TVector<TModelSplit> BinFeatures;
        ui32 EffectiveBinFeaturesBucketCount = 0;
    };

    struct TForApplyData {
        size_t UsedFloatFeaturesCount = 0;
        size_t UsedCatFeaturesCount = 0;
        size_t UsedTextFeaturesCount = 0;
        size_t UsedEmbeddingFeaturesCount = 0;
        size_t UsedEstimatedFeaturesCount = 0;
        size_t MinimalSufficientFloatFeaturesVectorSize = 0;
        size_t MinimalSufficientCatFeaturesVectorSize = 0;
        size_t MinimalSufficientTextFeaturesVectorSize = 0;
        size_t MinimalSufficientEmbeddingFeaturesVectorSize = 0;
        /**
         * List of all TModelCTR used in model
         */
        TVector<TModelCtr> UsedModelCtrs;

        //! Offset of first tree leaf in flat tree leafs array
        TVector<size_t> TreeFirstLeafOffsets;

        /**
         * List all unique CTR bases (feature combination + ctr type) in model
         * @return
         */
        TVector<TModelCtrBase> GetUsedModelCtrBases() const {
            THashSet<TModelCtrBase> ctrsSet;
            for (const auto& usedCtr : UsedModelCtrs) {
                ctrsSet.insert(usedCtr.Base);
            }
            TVector<TModelCtrBase> sortedBases(ctrsSet.begin(), ctrsSet.end());
            Sort(sortedBases.begin(), sortedBases.end());
            return sortedBases;
        }
    };

public:
    TModelTrees();
    TModelTrees(const TModelTrees& other) {
        *this = other;
    }

    TModelTrees& operator=(const TModelTrees& other) {
        if (this == &other) {
            return *this;
        }

        std::tie(
            ApproxDimension,
            CatFeatures,
            FloatFeatures,
            TextFeatures,
            EmbeddingFeatures,
            OneHotFeatures,
            CtrFeatures,
            EstimatedFeatures,
            ScaleAndBias,
            ModelTreeData
        )
        = std::forward_as_tuple(
            other.ApproxDimension,
            other.CatFeatures,
            other.FloatFeatures,
            other.TextFeatures,
            other.EmbeddingFeatures,
            other.OneHotFeatures,
            other.CtrFeatures,
            other.EstimatedFeatures,
            other.ScaleAndBias,
            other.ModelTreeData->Clone(IModelTreeData::ECloningPolicy::Default)
        );

        RepackedBins = other.RepackedBins;
        RuntimeData = other.RuntimeData;
        ApplyData = other.ApplyData;

        return *this;
    }

    bool operator==(const TModelTrees& other) const {
        return std::forward_as_tuple(
            ApproxDimension,
            GetModelTreeData()->GetTreeSplits(),
            GetModelTreeData()->GetTreeSizes(),
            GetModelTreeData()->GetTreeStartOffsets(),
            GetModelTreeData()->GetNonSymmetricStepNodes(),
            GetModelTreeData()->GetNonSymmetricNodeIdToLeafId(),
            GetModelTreeData()->GetLeafValues(),
            CatFeatures,
            FloatFeatures,
            TextFeatures,
            EmbeddingFeatures,
            OneHotFeatures,
            CtrFeatures,
            EstimatedFeatures,
            ScaleAndBias)
          == std::forward_as_tuple(
            other.ApproxDimension,
            other.GetModelTreeData()->GetTreeSplits(),
            other.GetModelTreeData()->GetTreeSizes(),
            other.GetModelTreeData()->GetTreeStartOffsets(),
            other.GetModelTreeData()->GetNonSymmetricStepNodes(),
            other.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId(),
            other.GetModelTreeData()->GetLeafValues(),
            other.CatFeatures,
            other.FloatFeatures,
            other.TextFeatures,
            other.EmbeddingFeatures,
            other.OneHotFeatures,
            other.CtrFeatures,
            other.EstimatedFeatures,
            other.ScaleAndBias);
    }

    bool operator!=(const TModelTrees& other) const {
        return !(*this == other);
    }

    bool IsOblivious() const {
        return GetModelTreeData()->GetNonSymmetricStepNodes().empty() && GetModelTreeData()->GetNonSymmetricNodeIdToLeafId().empty();
    }

    bool IsSolid() const;

    void ConvertObliviousToAsymmetric();

    /**
     * Method for oblivious trees serialization with repeated parts caching
     * @param serializer our caching flatbuffers serializator
     * @return offset in flatbuffer
     */
    flatbuffers::Offset<NCatBoostFbs::TModelTrees> FBSerialize(
        TModelPartsCachingSerializer& serializer) const;

    /**
     * Deserialize from flatbuffers object
     * @param fbObj
     */
    void FBDeserializeOwning(const NCatBoostFbs::TModelTrees* fbObj);
    void FBDeserializeNonOwning(const NCatBoostFbs::TModelTrees* fbObj);

    /**
     * Internal usage only.
     * Insert binary conditions tree with proper TreeSizes and TreeStartOffsets modification.
     * @param binSplits
     */
    void AddBinTree(const TVector<int>& binSplits);

    void SetTreeSplits(const TVector<int> &v) {
        ModelTreeData->SetTreeSplits(v);
    }

    void SetTreeSizes(const TVector<int> &v) {
        ModelTreeData->SetTreeSizes(v);
    }

    void SetTreeStartOffsets(const TVector<int> &v) {
        ModelTreeData->SetTreeStartOffsets(v);
    }

    void SetNonSymmetricStepNodes(const TVector<TNonSymmetricTreeStepNode> &v) {
        ModelTreeData->SetNonSymmetricStepNodes(v);
    }

    void SetNonSymmetricNodeIdToLeafId(const TVector<ui32> &v) {
        ModelTreeData->SetNonSymmetricNodeIdToLeafId(v);
    }

    void SetLeafValues(const TVector<double> &v) {
        ModelTreeData->SetLeafValues(v);
    }

    void SetLeafWeights(const TVector<double> &v) {
        ModelTreeData->SetLeafWeights(v);
    }

    size_t GetTreeCount() const {
        return GetModelTreeData()->GetTreeSizes().size();
    }

    size_t GetDimensionsCount() const noexcept {
        return ApproxDimension;
    }

    const THolder<IModelTreeData>& GetModelTreeData() const noexcept {
        return ModelTreeData;
    }

    TConstArrayRef<TCatFeature> GetCatFeatures() const noexcept {
        return TConstArrayRef<TCatFeature>(CatFeatures.begin(), CatFeatures.end());
    }

    TConstArrayRef<TFloatFeature> GetFloatFeatures() const noexcept {
        return TConstArrayRef<TFloatFeature>(FloatFeatures.begin(), FloatFeatures.end());
    }

    TConstArrayRef<TOneHotFeature> GetOneHotFeatures() const noexcept {
        return TConstArrayRef<TOneHotFeature>(OneHotFeatures.begin(), OneHotFeatures.end());
    }

    TConstArrayRef<TCtrFeature> GetCtrFeatures() const noexcept {
        return TConstArrayRef<TCtrFeature>(CtrFeatures.begin(), CtrFeatures.end());
    }

    TConstArrayRef<TTextFeature> GetTextFeatures() const noexcept {
        return TConstArrayRef<TTextFeature>(TextFeatures.begin(), TextFeatures.end());
    }

    TConstArrayRef<TEmbeddingFeature> GetEmbeddingFeatures() const noexcept {
        return TConstArrayRef<TEmbeddingFeature>(EmbeddingFeatures.begin(), EmbeddingFeatures.end());
    }

    TConstArrayRef<TEstimatedFeature> GetEstimatedFeatures() const noexcept {
        return TConstArrayRef<TEstimatedFeature>(EstimatedFeatures.begin(), EstimatedFeatures.end());
    }

    void SetApproxDimension(int approxDimension) {
        ApproxDimension = approxDimension;
        SetScaleAndBias({ScaleAndBias.Scale, TVector<double>(ApproxDimension, 0)});
    }

    void ClearLeafWeights();

    void SetCatFeatures(const TVector<TCatFeature>& catFeatures) {
        CatFeatures = catFeatures;
    }

    void SetFloatFeatures(const TVector<TFloatFeature>& floatFeatures) {
        FloatFeatures = floatFeatures;
    }

    void SetTextFeatures(const TVector<TTextFeature>& textFeatures) {
        TextFeatures = textFeatures;
    }

    void SetEmbeddingFeatures(const TVector<TEmbeddingFeature>& embeddingFeatures) {
        EmbeddingFeatures = embeddingFeatures;
    }

    void SetEstimatedFeatures(const TVector<TEstimatedFeature>& estimatedFeatures) {
        EstimatedFeatures = estimatedFeatures;
    }

    void ProcessSplitsSet(
        const TSet<TModelSplit>& modelSplitSet,
        const TVector<size_t>& floatFeaturesInternalIndexesMap,
        const TVector<size_t>& catFeaturesInternalIndexesMap,
        const TVector<size_t>& textFeaturesInternalIndexesMap,
        const TVector<size_t>& embeddingFeaturesInternalIndexesMap
    );

    void ApplyFeatureNames(const TVector<TString>& featureNames) {
        auto setFeatureName = [&] (TFeatureBase& feature) {
            size_t flatIndex = static_cast<size_t>(feature.Position.FlatIndex);
            CB_ENSURE(
                flatIndex < featureNames.size(),
                "Model has a feature with index " << flatIndex << " but provided features names size "
                << featureNames.size() << "is too small for it"
            );
            feature.FeatureId = featureNames[flatIndex];
        };

        for (TFloatFeature& feature : FloatFeatures) {
            setFeatureName(feature);
        }
        for (TCatFeature& feature : CatFeatures) {
            setFeatureName(feature);
        }
        for (TTextFeature& feature : TextFeatures) {
            setFeatureName(feature);
        }
        for (TEmbeddingFeature& feature : EmbeddingFeatures) {
            setFeatureName(feature);
        }
    }

    void AddFloatFeature(const TFloatFeature& floatFeature) {
        FloatFeatures.push_back(floatFeature);
    }

    void AddFloatFeatureBorder(const int featureId, const float border) {
        FloatFeatures[featureId].Borders.push_back(border);
    }

    void AddCatFeature(const TCatFeature& catFeature) {
        CatFeatures.push_back(catFeature);
    }

    void AddOneHotFeature(const TOneHotFeature& oneHotFeature) {
        OneHotFeatures.push_back(oneHotFeature);
    }

    void AddCtrFeature(const TCtrFeature& ctrFeature) {
        CtrFeatures.push_back(ctrFeature);
    }

    void AddTreeSplit(int treeSplit);
    void AddTreeSize(int treeSize);
    void AddLeafValue(double leafValue);
    void AddLeafWeight(double leafWeight);

    /**
     * Truncate oblivous trees to contain only trees from [begin; end) interval.
     * @param begin
     * @param end
     */
    void TruncateTrees(size_t begin, size_t end);

    /**
     * Drop unused float and categorical features from model
     */
     void DropUnusedFeatures();

    /**
     * Internal usage only. Updates UsedModelCtrs and BinFeatures vectors in RuntimeData to contain all
     *  features currently used in model.
     * Should be called after any modifications.
     */
    void UpdateRuntimeData();

    TAtomicSharedPtr<TForApplyData> GetApplyData() const {
        return ApplyData;
    }

    /**
     * List all binary features corresponding to binary feature indexes in trees
     * @return
     */
    TConstArrayRef<TModelSplit> GetBinFeatures() const {
        return RuntimeData->BinFeatures;
    }

    TConstArrayRef<TRepackedBin> GetRepackedBins() const {
        return *RepackedBins;
    }

    const double* GetFirstLeafPtrForTree(size_t treeIdx) const {
        auto applyData = GetApplyData();
        return &ModelTreeData->GetLeafValues()[applyData->TreeFirstLeafOffsets[treeIdx]];
    }

    size_t GetNumFloatFeatures() const {
        if (FloatFeatures.empty()) {
            return 0;
        } else {
            return static_cast<size_t>(FloatFeatures.back().Position.Index) + 1;
        }
    }

    size_t GetNumCatFeatures() const {
        if (CatFeatures.empty()) {
            return 0;
        } else {
            return static_cast<size_t>(CatFeatures.back().Position.Index) + 1;
        }
    }

    size_t GetNumTextFeatures() const {
        if (TextFeatures.empty()) {
            return 0;
        } else {
            return static_cast<size_t>(TextFeatures.back().Position.Index) + 1;
        }
    }

    size_t GetNumEmbeddingFeatures() const {
        if (EmbeddingFeatures.empty()) {
            return 0;
        } else {
            return static_cast<size_t>(EmbeddingFeatures.back().Position.Index) + 1;
        }
    }

    size_t GetBinaryFeaturesFullCount() const {
        return GetBinFeatures().size();
    }

    ui32 GetEffectiveBinaryFeaturesBucketsCount() const {
        return RuntimeData->EffectiveBinFeaturesBucketCount;
    }

    size_t GetFlatFeatureVectorExpectedSize() const {
        return (size_t)Max(
            CatFeatures.empty() ? 0 : CatFeatures.back().Position.FlatIndex + 1,
            FloatFeatures.empty() ? 0 : FloatFeatures.back().Position.FlatIndex + 1,
            TextFeatures.empty() ? 0 : TextFeatures.back().Position.FlatIndex + 1,
            EmbeddingFeatures.empty() ? 0 : EmbeddingFeatures.back().Position.FlatIndex + 1
        );
    }

    TVector<ui32> GetTreeLeafCounts() const;

    const TScaleAndBias& GetScaleAndBias() const {
        return ScaleAndBias;
    }

    void SetScaleAndBias(const TScaleAndBias&);

private:
    void DeserializeFeatures(const NCatBoostFbs::TModelTrees* fbObj);

    void SetScaleAndBias(const NCatBoostFbs::TModelTrees* fbObj);

    void CalcBinFeatures();
    void CalcForApplyData() {
        ApplyData = MakeAtomicShared<TForApplyData>();
        ProcessFloatFeatures();
        ProcessCatFeatures();
        ProcessTextFeatures();
        ProcessEmbeddingFeatures();
        ProcessEstimatedFeatures();
        CalcUsedModelCtrs();
        CalcFirstLeafOffsets();
    }
    void CalcUsedModelCtrs();
    void CalcFirstLeafOffsets();
    void ProcessFloatFeatures();
    void ProcessCatFeatures();
    void ProcessTextFeatures();
    void ProcessEstimatedFeatures();
    void ProcessEmbeddingFeatures();
private:
    //! Number of classes in model, in most cases equals to 1.
    int ApproxDimension = 1;

    THolder<IModelTreeData> ModelTreeData;

    //! Categorical features, used in model in OneHot conditions or/and in CTR feature combinations
    TVector<TCatFeature> CatFeatures;

    static_assert(ESplitType::FloatFeature < ESplitType::OneHotFeature
                  && ESplitType::OneHotFeature < ESplitType::OnlineCtr,
                  "ESplitType should represent bin feature order in model");

    //! Float features used in model
    TVector<TFloatFeature> FloatFeatures;
    //! One hot encoded features used in model
    TVector<TOneHotFeature> OneHotFeatures;
    //! CTR features used in model
    TVector<TCtrFeature> CtrFeatures;

    //! Text features used in model
    TVector<TTextFeature> TextFeatures;

    //! Embedding features used in model
    TVector<TEmbeddingFeature> EmbeddingFeatures;

    //! Computed on text and embedding features used in model
    TVector<TEstimatedFeature> EstimatedFeatures;

    //! For computing final formula result as `Scale * sumTrees + Bias`
    TScaleAndBias ScaleAndBias;

    TAtomicSharedPtr<TRuntimeData> RuntimeData;
    TAtomicSharedPtr<TForApplyData> ApplyData;

    /**
    * This vector contains ui32 that contains such information:
    * |     ui16     |   ui8   |   ui8  |
    * | featureIndex | xorMask |splitIdx| (e.g. featureIndex << 16 + xorMask << 8 + splitIdx )
    *
    * We use this layout to speed up model apply - we only need to store one byte for each float, ctr or
    *  one hot feature.
    */

    static_assert(sizeof(TRepackedBin) == 4, "");

    mutable NCB::TMaybeOwningConstArrayHolder<TRepackedBin> RepackedBins;
};

class TCOWTreeWrapper {
public:
    const TModelTrees& operator*() const noexcept {
        return *Trees;
    }
    const TModelTrees* operator->() const noexcept {
        return Trees.Get();
    }

    const TModelTrees* Get() const noexcept {
        return Trees.Get();
    }

    TModelTrees* GetMutable() {
        if (Trees.RefCount() > 1) {
            Trees = MakeAtomicShared<TModelTrees>(*Trees);
        }
        return Trees.Get();
    }
private:
    TAtomicSharedPtr<TModelTrees> Trees = MakeAtomicShared<TModelTrees>();
};

/*!
 * \brief Full model class - contains all the data for model evaluation
 *
 * This class contains oblivious trees data, key-value dictionary for model metadata storage and CtrProvider
 *  holder.
 */
class TFullModel {
public:
    using TFeatureLayout = NCB::NModelEvaluation::TFeatureLayout;
public:
    TCOWTreeWrapper ModelTrees;
    /**
     * Model information key-value storage.
     */
    THashMap<TString, TString> ModelInfo;
    TIntrusivePtr<ICtrProvider> CtrProvider;
    TIntrusivePtr<NCB::TTextProcessingCollection> TextProcessingCollection;
    TIntrusivePtr<NCB::TEmbeddingProcessingCollection> EmbeddingProcessingCollection;
private:
    EFormulaEvaluatorType FormulaEvaluatorType = EFormulaEvaluatorType::CPU;
    TAdaptiveLock CurrentEvaluatorLock;
    mutable NCB::NModelEvaluation::TModelEvaluatorPtr Evaluator;
public:
    void InitNonOwning(const void* binaryBuffer, size_t dataSize);

    static TVector<EFormulaEvaluatorType> GetSupportedEvaluatorTypes();

    void SetEvaluatorType(EFormulaEvaluatorType evaluatorType) {
        with_lock(CurrentEvaluatorLock) {
            if (FormulaEvaluatorType != evaluatorType) {
                Evaluator = NCB::NModelEvaluation::CreateEvaluator(evaluatorType, *this); // we can fail here
                FormulaEvaluatorType = evaluatorType;
            }
        }
    }

    NCB::NModelEvaluation::TConstModelEvaluatorPtr GetCurrentEvaluator() const {
        with_lock(CurrentEvaluatorLock) {
            if (!Evaluator) {
                Evaluator = NCB::NModelEvaluation::CreateEvaluator(FormulaEvaluatorType, *this);
            }
            return Evaluator;
        }
    }

    void SetPredictionType(NCB::NModelEvaluation::EPredictionType predictionType) const {
        with_lock(CurrentEvaluatorLock) {
            if (!Evaluator) {
                Evaluator = NCB::NModelEvaluation::CreateEvaluator(FormulaEvaluatorType, *this);
            }
            Evaluator->SetPredictionType(predictionType);
        }
    }

    EFormulaEvaluatorType GetEvaluatorType() const {
        return FormulaEvaluatorType;
    }

    bool operator==(const TFullModel& other) const {
        return *ModelTrees == *other.ModelTrees;
    }

    bool operator!=(const TFullModel& other) const {
        return !(*this == other);
    }

    //TODO(kirillovs): get rid of this method
    void Swap(TFullModel& other) {
        with_lock(CurrentEvaluatorLock) {
            with_lock(other.CurrentEvaluatorLock) {
                DoSwap(ModelTrees, other.ModelTrees);
                DoSwap(ModelInfo, other.ModelInfo);
                DoSwap(CtrProvider, other.CtrProvider);
                DoSwap(FormulaEvaluatorType, other.FormulaEvaluatorType);
                DoSwap(Evaluator, other.Evaluator);
            }
        }
        DoSwap(TextProcessingCollection, other.TextProcessingCollection);
        DoSwap(EmbeddingProcessingCollection, other.EmbeddingProcessingCollection);
    }

    /**
     * Check whether model contains categorical features in OneHot conditions and/or CTR feature combinations
     */
    bool HasCategoricalFeatures() const {
        return GetUsedCatFeaturesCount() != 0;
    }

    /**
     * Check wheter model contains text features
     */
    bool HasTextFeatures() const {
        return GetUsedTextFeaturesCount() != 0;
    }

    bool HasEmbeddingFeatures() const {
        return GetUsedEmbeddingFeaturesCount() != 0;
    }

    /**
     * @return Number of trees in model.
     */
    size_t GetTreeCount() const {
        return ModelTrees->GetTreeCount();
    }

    /**
     * @return Number of dimensions in model.
     */
    size_t GetDimensionsCount() const noexcept {
        return ModelTrees->GetDimensionsCount();
    }

    /**
     * Truncate trees to contain only trees from [begin; end) interval.
     * @param begin
     * @param end
     */
    void Truncate(size_t begin, size_t end) {
        auto applyData = ModelTrees->GetApplyData();
        ModelTrees.GetMutable()->TruncateTrees(begin, end);
        if (CtrProvider) {
            CtrProvider->DropUnusedTables(applyData->GetUsedModelCtrBases());
        }
        if (begin > 0) {
            SetScaleAndBias({GetScaleAndBias().Scale, {}});
        }
        UpdateDynamicData();
    }

    /**
     * @return Minimal float features vector length sufficient for this model
     */
    size_t GetMinimalSufficientFloatFeaturesVectorSize() const {
        auto applyData = ModelTrees->GetApplyData();
        return applyData->MinimalSufficientFloatFeaturesVectorSize;
    }
    /**
     * @return Number of float features that are really used in trees
     */
    size_t GetUsedFloatFeaturesCount() const {
        auto applyData = ModelTrees->GetApplyData();
        return applyData->UsedFloatFeaturesCount;
    }

    /**
     * @return Number of text features that are really used in trees
     */
    size_t GetUsedTextFeaturesCount() const {
        auto applyData = ModelTrees->GetApplyData();
        return applyData->UsedTextFeaturesCount;
    }

    size_t GetUsedEmbeddingFeaturesCount() const {
        return ModelTrees->GetApplyData()->UsedEmbeddingFeaturesCount;
    }

    /**
     * @return Expected float features vector length for this model
     */
    size_t GetNumFloatFeatures() const {
        return ModelTrees->GetNumFloatFeatures();
    }

    /**
     * @return Expected categorical features vector length for this model
     */
    size_t GetMinimalSufficientCatFeaturesVectorSize() const {
        auto applyData = ModelTrees->GetApplyData();
        return applyData->MinimalSufficientCatFeaturesVectorSize;
    }
    /**
    * @return Number of float features that are really used in trees
    */
    size_t GetUsedCatFeaturesCount() const {
        auto applyData = ModelTrees->GetApplyData();
        return applyData->UsedCatFeaturesCount;
    }

    /**
     * @return Expected categorical features vector length for this model
     */
    size_t GetNumCatFeatures() const {
        return ModelTrees->GetNumCatFeatures();
    }

    /**
    * @return Expected text features vector length for this model
    */
    size_t GetNumTextFeatures() const {
        return ModelTrees->GetNumTextFeatures();
    }

    /**
    * @return Expected embeddings features vector length for this model
    */
    size_t GetNumEmbeddingFeatures() const {
        return ModelTrees->GetNumEmbeddingFeatures();
    }

    /**
     * Check whether model trees are oblivious
     */
    bool IsOblivious() const {
        return ModelTrees->IsOblivious();
    }

    /**
     * Serialize model to stream
     * @param s IOutputStream ptr
     */
    void Save(IOutputStream* s) const;

    /**
     * Deserialize model from stream
     * @param s IInputStream ptr
     */
    void Load(IInputStream* s);

    //! Check if TFullModel instance has valid CTR provider.
    // If no ctr features present it will return true
    bool HasValidCtrProvider() const {
        auto applyData = ModelTrees->GetApplyData();
        if (!CtrProvider) {
            return applyData->UsedModelCtrs.empty();
        }
        return CtrProvider->HasNeededCtrs(applyData->UsedModelCtrs);
    }

    //! Check if TFullModel instance has valid Text processing collection
    bool HasValidTextProcessingCollection() const {
        return (bool) TextProcessingCollection;
    }

    bool HasValidEmbeddingProcessingCollection() const {
        return (bool) EmbeddingProcessingCollection;
    }

    //! Get normalization parameters used to compute final formula from sum of trees
    const TScaleAndBias& GetScaleAndBias() const {
        return ModelTrees->GetScaleAndBias();
    }

    //! Set normalization parameters for computing final formula from sum of trees
    void SetScaleAndBias(const TScaleAndBias& scaleAndBias) {
        ModelTrees.GetMutable()->SetScaleAndBias(scaleAndBias);
        with_lock(CurrentEvaluatorLock) {
            Evaluator.Reset();
        }
    }

    /**
     * Special interface for model evaluation on transposed dataset layout
     * @param[in] transposedFeatures transposed flat features vector. First dimension is feature index,
     *  second dimension is object index.
     * If feature is categorical, we do reinterpret cast from float to int.
     * @param[in] treeStart Index of first tree in model to start evaluation
     * @param[in] treeEnd Index of tree after the last tree in model to evaluate. F.e. if you want to evaluate
     *  trees 2..5 use treeStart = 2, treeEnd = 6
     * @param[out] results Flat double vector with indexation [objectIndex * ApproxDimension + classId].
     * For single class models it is just [objectIndex]
     */
    void CalcFlatTransposed(
        TConstArrayRef<TConstArrayRef<float>> transposedFeatures,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const;

    /**
     * Special interface for model evaluation on flat feature vectors. Flat here means that float features and
     *  categorical feature are in the same float array.
     * @param[in] features vector of flat features array reference. First dimension is object index, second
     *  dimension is feature index.
     * If feature is categorical, we do reinterpret cast from float to int.
     * @param[in] treeStart Index of first tree in model to start evaluation
     * @param[in] treeEnd Index of tree after the last tree in model to evaluate. F.e. if you want to evaluate
     *  trees 2..5 use treeStart = 2, treeEnd = 6
     * @param[out] results Flat double vector with indexation [objectIndex * ApproxDimension + classId].
     * For single class models it is just [objectIndex]
     */
    void CalcFlat(
        TConstArrayRef<TConstArrayRef<float>> features,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const;

    /**
     * Call CalcFlat on all model trees
     * @param features
     * @param results
     */
    void CalcFlat(
        TConstArrayRef<TConstArrayRef<float>> features,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        CalcFlat(features, 0, GetTreeCount(), results, featureInfo);
    }

    /**
     * Call CalcFlat on all model trees
     * @param features
     * @param results
     */
    void CalcFlat(
        TConstArrayRef<TVector<float>> features,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        TVector<TConstArrayRef<float>> featureRefs{features.begin(), features.end()};
        CalcFlat(featureRefs, results, featureInfo);
    }

    /**
     * Call CalcFlatTransposed on all model trees
     * @param features
     * @param results
     */
    void CalcFlatTransposed(
        TConstArrayRef<TConstArrayRef<float>> features,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        CalcFlatTransposed(features, 0, GetTreeCount(), results, featureInfo);
    }

    /**
     * Call CalcFlatTransposed on all model trees
     * @param features
     * @param results
     */
    void CalcFlatTransposed(
        TConstArrayRef<TVector<float>> features,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        TVector<TConstArrayRef<float>> featureRefs{features.begin(), features.end()};
        CalcFlatTransposed(featureRefs, results, featureInfo);
    }

    /**
     * Same as CalcFlat method but for one object
     * @param[in] features flat features array reference. First dimension is object index, second dimension is
     *  feature index.
     * If feature is categorical, we do reinterpret cast from float to int.
     * @param[in] treeStart Index of first tree in model to start evaluation
     * @param[in] treeEnd Index of tree after the last tree in model to evaluate. F.e. if you want to evaluate
     *  trees 2..5 use treeStart = 2, treeEnd = 6
     * @param[out] results double vector with indexation [classId].
     */
    void CalcFlatSingle(
        TConstArrayRef<float> features,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const;

    /**
     * CalcFlatSingle on all trees in the model
     * @param[in] features flat features array reference. First dimension is object index, second dimension is
     *  feature index.
     * If feature is categorical, we do reinterpret cast from float to int.
     * @param[out] results double vector with indexation [classId].
     */
    void CalcFlatSingle(
        TConstArrayRef<float> features,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        CalcFlatSingle(features, 0, GetTreeCount(), results, featureInfo);
    }

    /**
     * Shortcut for CalcFlatSingle
     */
    void CalcFlat(TConstArrayRef<float> features, TArrayRef<double> result) const {
        CalcFlatSingle(features, result);
    }

    /**
     * Evaluate raw formula predictions on user data. Uses model trees for interval [treeStart, treeEnd)
     * @param[in] floatFeatures
     * @param[in] catFeatures hashed cat feature values
     * @param[in] treeStart
     * @param[in] treeEnd
     * @param[out] results results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TConstArrayRef<int>> catFeatures,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr) const;

    /**
     * Evaluate raw formula predictions on user data. Uses all model trees
     * @param floatFeatures
     * @param catFeatures hashed cat feature values
     * @param results results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TConstArrayRef<int>> catFeatures,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        Calc(floatFeatures, catFeatures, 0, GetTreeCount(), results, featureInfo);
    }

    /**
     * Evaluate raw formula predictions on user data. Uses model trees for interval [treeStart, treeEnd)
     * @param[in] floatFeatures
     * @param[in] catFeatures hashed cat feature values
     * @param[in] textFeatures
     * @param[in] embeddingFeatures
     * @param[in] treeStart
     * @param[in] treeEnd
     * @param[out] results results indexation is [objectIndex * ApproxDimension + classId]
     */
    void CalcWithHashedCatAndTextAndEmbeddings(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TConstArrayRef<int>> catFeatures,
        TConstArrayRef<TVector<TStringBuf>> textFeatures,
        TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr) const;

    /**
     * Evaluate raw formula predictions on user data. Uses all model trees
     * @param floatFeatures
     * @param catFeatures hashed cat feature values
     * @param textFeatures
     * @param[in] embeddingFeatures
     * @param results results indexation is [objectIndex * ApproxDimension + classId]
     */
    void CalcWithHashedCatAndTextAndEmbeddings(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TConstArrayRef<int>> catFeatures,
        TConstArrayRef<TVector<TStringBuf>> textFeatures,
        TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        CalcWithHashedCatAndTextAndEmbeddings(floatFeatures, catFeatures, textFeatures, embeddingFeatures, 0, GetTreeCount(), results, featureInfo);
    }

    /**
     * Evaluate raw formula prediction for one object. Uses all model trees
     * @param floatFeatures
     * @param catFeatures
     * @param result indexation is [classId]
     */
    void Calc(
        TConstArrayRef<float> floatFeatures,
        TConstArrayRef<int> catFeatures,
        TArrayRef<double> result,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        const TConstArrayRef<float> floatFeaturesArray[] = {floatFeatures};
        const TConstArrayRef<int> catFeaturesArray[] = {catFeatures};
        Calc(floatFeaturesArray, catFeaturesArray, result, featureInfo);
    }

    /**
     * Evaluate raw formula predictions for objects. Uses model trees from interval [treeStart, treeEnd)
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @param treeStart
     * @param treeEnd
     * @param results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TStringBuf>> catFeatures,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const;

    /**
     * Evaluate raw formula predictions for objects. Uses all model trees.
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @param results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TStringBuf>> catFeatures,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        Calc(floatFeatures, catFeatures, 0, GetTreeCount(), results, featureInfo);
    }

    /**
     * Evaluate raw formula predictions for objects. Uses all model trees.
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @param treeStart
     * @param treeEnd
     * @param textFeatures vector of vector of TStringBuf with features containing text as strings
     * @param results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TStringBuf>> catFeatures,
        TConstArrayRef<TVector<TStringBuf>> textFeatures,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const;

    /**
     * Evaluate raw formula predictions for objects. Uses all model trees.
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @param textFeatures vector of vector of TStringBuf with features containing text as strings
     * @param results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TStringBuf>> catFeatures,
        TConstArrayRef<TVector<TStringBuf>> textFeatures,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        Calc(floatFeatures, catFeatures, textFeatures, 0, GetTreeCount(), results, featureInfo);
    }

    /**
     * Evaluate raw formula predictions for objects. Uses all model trees.
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @param textFeatures vector of vector of TStringBuf with features containing text as strings
     * @param embeddingFeatures vector of vector of vectors of embeddings
     * @param treeStart
     * @param treeEnd
     * @param results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TStringBuf>> catFeatures,
        TConstArrayRef<TVector<TStringBuf>> textFeatures,
        TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const;

    /**
     * Evaluate raw formula predictions for objects. Uses all model trees.
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @param textFeatures vector of vector of TStringBuf with features containing text as strings
     * @param embeddingFeatures vector of vector of vectors of embeddings
     * @param results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TStringBuf>> catFeatures,
        TConstArrayRef<TVector<TStringBuf>> textFeatures,
        TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
        TArrayRef<double> results,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        Calc(floatFeatures, catFeatures, textFeatures, embeddingFeatures, 0, GetTreeCount(), results, featureInfo);
    }

    /**
     * Truncate model to contain only trees from [begin; end) interval.
     * @param begin
     * @param end
     * @return model copy that contains only needed trees
     */
    TFullModel CopyTreeRange(size_t begin, size_t end) const {
        TFullModel result = *this;
        if (CtrProvider) {
            result.CtrProvider = CtrProvider->Clone();
        }
        result.Truncate(begin, end);
        return result;
    }

    /**
     * Evaluate indexes of leafs at which object are mapped by trees from interval [treeStart, treeEnd).
     * @param floatFeatures
     * @param catFeatures vector of TStringBuf with categorical features strings
     * @param treeStart
     * @param treeEnd
     * @return indexes; size should be equal to (treeEnd - treeStart).
     */
    void CalcLeafIndexesSingle(
        TConstArrayRef<float> floatFeatures,
        TConstArrayRef<TStringBuf> catFeatures,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<ui32> indexes,
        const TFeatureLayout* featureInfo = nullptr
    ) const;

    /**
     * Evaluate indexes of leafs at which object are mapped by all trees of the model.
     * @param floatFeatures
     * @param catFeatures vector of TStringBuf with categorical features strings
     * @return indexes; size should be equal to number of trees in the model.
     */
    void CalcLeafIndexesSingle(
        TConstArrayRef<float> floatFeatures,
        TConstArrayRef<TStringBuf> catFeatures,
        TArrayRef<ui32> indexes,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        CalcLeafIndexesSingle(floatFeatures, catFeatures, 0, GetTreeCount(), indexes, featureInfo);
    }

    /**
     * Evaluate indexes of leafs at which objects are mapped by trees from interval [treeStart, treeEnd).
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @param treeStart
     * @param treeEnd
     * @return indexes; indexation is [objectIndex * (treeEnd -  treeStrart) + treeIndex]
     */
    void CalcLeafIndexes(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<ui32> indexes,
        const TFeatureLayout* featureInfo = nullptr
    ) const;

    /**
     * Evaluate indexes of leafs at which objects are mapped by all trees of the model.
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @return indexes; indexation is [objectIndex * treeCount + treeIndex]
     */
    void CalcLeafIndexes(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
        TArrayRef<ui32> indexes,
        const TFeatureLayout* featureInfo = nullptr
    ) const {
        CalcLeafIndexes(floatFeatures, catFeatures, 0, GetTreeCount(), indexes, featureInfo);
    }

    /**
     * Get the name of optimized objective used to train the model.
     * @return the name, or empty string if the model does not have this information
     */
    TString GetLossFunctionName() const;

    /**
     * Get the probability threshold for binary classification to separate classes.
     * @return the value is stored in `binclass_probability_threshold` metadata or 0.5 as default value.
     */
    double GetBinClassProbabilityThreshold() const;

    /**
     * Get the logit threshold for binary classification to separate classes.
     * @return Logit(GetBinClassProbabilityThreshold())
     */
    double GetBinClassLogitThreshold() const;

    /**
     * Get typed class labels than can be predicted.
     *
     * @return Vector of typed class labels corresponding to approx dimension if the model can be used for
     *    classification or empty vector otherwise.
     *    Possible value types are Integer, Float or String
     */
    TVector<NJson::TJsonValue> GetModelClassLabels() const;

    /**
     * Internal usage only.
     * Updates indexes in CTR provider and recalculates runtime data in Oblivious trees after model
     *  modifications.
     */
    void UpdateDynamicData();

    /**
     * Internal usage only.
     * Update indexes between TextProcessingCollection and Estimated features in ModelTrees
     */
    void UpdateEstimatedFeaturesIndices(TVector<TEstimatedFeature>&& newEstimatedFeatures);

    bool IsPosteriorSamplingModel() const;

    float GetActualShrinkCoef() const;

private:
    void DefaultFullModelInit(const NCatBoostFbs::TModelCore* fbModelCore);
};

void OutputModel(const TFullModel& model, TStringBuf modelFile);
void OutputModel(const TFullModel& model, IOutputStream* out);

bool IsDeserializableModelFormat(EModelType format);

TFullModel ReadModel(const TString& modelFile, EModelType format = EModelType::CatboostBinary);
TFullModel ReadModel(
    const void* binaryBuffer,
    size_t binaryBufferSize,
    EModelType format = EModelType::CatboostBinary);
TFullModel ReadZeroCopyModel(const void* binaryBuffer, size_t binaryBufferSize);

/**
 * Serialize model to string
 * @param model
 * @return
 */
TString SerializeModel(const TFullModel& model);

/**
 * Deserialize model from a memory buffer
 * @param serializedModel
 * @return
 */
TFullModel DeserializeModel(TMemoryInput serializedModel);

/**
 * Deserialize model from a string
 * @param serializedModel
 * @return
 */
TFullModel DeserializeModel(const TString& serializedModel);

TVector<TString> GetModelUsedFeaturesNames(const TFullModel& model);

TVector<size_t> GetModelCatFeaturesIndices(const TFullModel& model);

TVector<size_t> GetModelFloatFeaturesIndices(const TFullModel& model);

TVector<size_t> GetModelTextFeaturesIndices(const TFullModel& model);

TVector<size_t> GetModelEmbeddingFeaturesIndices(const TFullModel& model);

void SetModelExternalFeatureNames(const TVector<TString>& featureNames, TFullModel* model);

TFullModel SumModels(
    const TVector<const TFullModel*> modelVector,
    const TVector<double>& weights,
    const TVector<TString>& modelParamsPrefixes = TVector<TString>(), // can be empty - in this case default prefixes will be used
    ECtrTableMergePolicy ctrMergePolicy = ECtrTableMergePolicy::IntersectingCountersAverage);

void SaveModelBorders(
    const TString& file,
    const TFullModel& model);

THashMap<int, TFloatFeature::ENanValueTreatment> GetNanTreatments(const TFullModel& model);
