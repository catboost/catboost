#pragma once

#include "ctr_provider.h"
#include "features.h"
#include "online_ctr.h"
#include "split.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/flatbuffers/model.fbs.h>
#include <catboost/libs/options/enums.h>

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
};


// If selected diff is 0 we are in the last node in path
struct TNonSymmetricTreeStepNode {
    static constexpr ui16 InvalidDiff = Max<ui16>();
    static constexpr ui16 TerminalMarker = 0;

    ui16 LeftSubtreeDiff = InvalidDiff;
    ui16 RightSubtreeDiff = InvalidDiff;

    static TNonSymmetricTreeStepNode TerminalNodeSteps() {
        return TNonSymmetricTreeStepNode{0, 0};
    }

    TNonSymmetricTreeStepNode& operator=(const NCatBoostFbs::TNonSymmetricTreeStepNode* stepNode) {
        LeftSubtreeDiff = stepNode->LeftSubtreeDiff();
        RightSubtreeDiff = stepNode->RightSubtreeDiff();
        return *this;
    }

    bool operator==(const TNonSymmetricTreeStepNode& other) const {
        return std::tie(LeftSubtreeDiff, RightSubtreeDiff)
            == std::tie(other.LeftSubtreeDiff, other.RightSubtreeDiff);
    }
};

// TODO(kirillovs): rename to TModelTrees after adding non symmetric trees support
struct TObliviousTrees {
public:
    /**
     * This structure stores model runtime data. Should be kept up to date
     */
    struct TRuntimeData {
        size_t UsedFloatFeaturesCount = 0;
        size_t UsedCatFeaturesCount = 0;
        size_t MinimalSufficientFloatFeaturesVectorSize = 0;
        size_t MinimalSufficientCatFeaturesVectorSize = 0;
        /**
         * List of all TModelCTR used in model
         */
        TVector<TModelCtr> UsedModelCtrs;
        /**
         * List of all binary with indexes corresponding to TreeSplits values
         */
        TVector<TModelSplit> BinFeatures;

        /**
        * This vector contains ui32 that contains such information:
        * |     ui16     |   ui8   |   ui8  |
        * | featureIndex | xorMask |splitIdx| (e.g. featureIndex << 16 + xorMask << 8 + splitIdx )
        *
        * We use this layout to speed up model apply - we only need to store one byte for each float, ctr or
        *  one hot feature.
        */

        static_assert(sizeof(TRepackedBin) == 4, "");

        TVector<TRepackedBin> RepackedBins;

        ui32 EffectiveBinFeaturesBucketCount = 0;

        //! Offset of first tree leaf in flat tree leafs array
        TVector<size_t> TreeFirstLeafOffsets;
    };

public:
    //! Number of classes in model, in most cases equals to 1.
    int ApproxDimension = 1;

    //! Split values
    TVector<int> TreeSplits;

    //! Tree sizes
    TVector<int> TreeSizes;

    //! Offset of first split in TreeSplits array
    TVector<int> TreeStartOffsets;

    //! Steps in a non-symmetric tree.
    //! If at least one diff in a step node is zero, it's a terminal node and has a value.
    //! If both diffs are zero, the corresponding split condition (in the RepackedBins vector) may be invalid.
    TVector<TNonSymmetricTreeStepNode> NonSymmetricStepNodes;

    //! Holds a value index (in the LeafValues vector) for each terminal node in a non-symmetric tree.
    //! For multiclass models holds indexes for 0-class.
    TVector<ui32> NonSymmetricNodeIdToLeafId;

    //! Leaf values layout: [treeIndex][leafId * ApproxDimension + dimension]
    TVector<double> LeafValues;

    /**
     * Leaf Weights are sums of weights or group weights of samples from the learn dataset that go to that leaf.
     * This information can be absent (this vector will be empty) in some models:
     *   - Loaded from CoreML format
     *   - Old models trained on GPU (trained with catboost version < 0.9.x)
     *
     *  layout: [treeIndex][leafId]
     */
    TVector<TVector<double>> LeafWeights;

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

public:
    bool operator==(const TObliviousTrees& other) const {
        return std::tie(
            ApproxDimension,
            TreeSplits,
            TreeSizes,
            TreeStartOffsets,
            NonSymmetricStepNodes,
            NonSymmetricNodeIdToLeafId,
            LeafValues,
            CatFeatures,
            FloatFeatures,
            OneHotFeatures,
            CtrFeatures)
          == std::tie(
            other.ApproxDimension,
            other.TreeSplits,
            other.TreeSizes,
            other.TreeStartOffsets,
            other.NonSymmetricStepNodes,
            other.NonSymmetricNodeIdToLeafId,
            other.LeafValues,
            other.CatFeatures,
            other.FloatFeatures,
            other.OneHotFeatures,
            other.CtrFeatures);
    }

    bool operator!=(const TObliviousTrees& other) const {
        return !(*this == other);
    }

    bool IsOblivious() const {
        return NonSymmetricStepNodes.empty() && NonSymmetricNodeIdToLeafId.empty();
    }

    void ConvertObliviousToAsymmetric();

    /**
     * Method for oblivious trees serialization with repeated parts caching
     * @param serializer our caching flatbuffers serializator
     * @return offset in flatbuffer
     */
    flatbuffers::Offset<NCatBoostFbs::TObliviousTrees> FBSerialize(
        TModelPartsCachingSerializer& serializer) const;

    /**
     * Deserialize from flatbuffers object
     * @param fbObj
     */
    void FBDeserialize(const NCatBoostFbs::TObliviousTrees* fbObj) {
        ApproxDimension = fbObj->ApproxDimension();
        if (fbObj->TreeSplits()) {
            TreeSplits.assign(fbObj->TreeSplits()->begin(), fbObj->TreeSplits()->end());
        }
        if (fbObj->TreeSizes()) {
            TreeSizes.assign(fbObj->TreeSizes()->begin(), fbObj->TreeSizes()->end());
        }
        if (fbObj->TreeStartOffsets()) {
            TreeStartOffsets.assign(fbObj->TreeStartOffsets()->begin(), fbObj->TreeStartOffsets()->end());
        }

        if (fbObj->LeafValues()) {
            LeafValues.assign(fbObj->LeafValues()->begin(), fbObj->LeafValues()->end());
        }
        if (fbObj->NonSymmetricStepNodes()) {
            NonSymmetricStepNodes.resize(fbObj->NonSymmetricStepNodes()->size());
            std::copy(
                fbObj->NonSymmetricStepNodes()->begin(),
                fbObj->NonSymmetricStepNodes()->end(),
                NonSymmetricStepNodes.begin()
            );
        }
        if (fbObj->NonSymmetricNodeIdToLeafId()) {
            NonSymmetricNodeIdToLeafId.assign(
                fbObj->NonSymmetricNodeIdToLeafId()->begin(), fbObj->NonSymmetricNodeIdToLeafId()->end()
            );
        }

#define FBS_ARRAY_DESERIALIZER(var) \
        if (fbObj->var()) {\
            var.resize(fbObj->var()->size());\
            for (size_t i = 0; i < fbObj->var()->size(); ++i) {\
                var[i].FBDeserialize(fbObj->var()->Get(i));\
            }\
        }
        FBS_ARRAY_DESERIALIZER(CatFeatures)
        FBS_ARRAY_DESERIALIZER(FloatFeatures)
        FBS_ARRAY_DESERIALIZER(OneHotFeatures)
        FBS_ARRAY_DESERIALIZER(CtrFeatures)
#undef FBS_ARRAY_DESERIALIZER
        if (fbObj->LeafWeights() && fbObj->LeafWeights()->size() > 0) {
            if (IsOblivious()) {
                LeafWeights.resize(TreeSizes.size());
                CB_ENSURE(fbObj->LeafWeights()->size() * ApproxDimension == LeafValues.size(),
                          "Bad leaf weights count: " << fbObj->LeafWeights()->size());
                auto leafValIter = fbObj->LeafWeights()->begin();
                for (size_t treeId = 0; treeId < TreeSizes.size(); ++treeId) {
                    const auto treeLeafCout = (1 << TreeSizes[treeId]);
                    LeafWeights[treeId].assign(leafValIter, leafValIter + treeLeafCout);
                    leafValIter += treeLeafCout;
                }
            } else {
                LeafWeights.resize(1);
                LeafWeights[0].assign(fbObj->LeafWeights()->begin(), fbObj->LeafWeights()->end());
            }
        }
    }

    /**
     * Internal usage only.
     * Insert binary conditions tree with proper TreeSizes and TreeStartOffsets modification.
     * @param binSplits
     */
    void AddBinTree(const TVector<int>& binSplits) {
        Y_ASSERT(TreeSizes.size() == TreeStartOffsets.size() && (TreeSplits.empty() == TreeSizes.empty()));
        TreeSplits.insert(TreeSplits.end(), binSplits.begin(), binSplits.end());
        if (TreeStartOffsets.empty()) {
            TreeStartOffsets.push_back(0);
        } else {
            TreeStartOffsets.push_back(TreeStartOffsets.back() + TreeSizes.back());
        }
        TreeSizes.push_back(binSplits.ysize());
    }

    size_t GetTreeCount() const {
        return TreeSizes.size();
    }

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
    void UpdateRuntimeData() const;
    /**
     * List of all CTRs in model
     * @return
     */
    const TVector<TModelCtr>& GetUsedModelCtrs() const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return RuntimeData->UsedModelCtrs;
    }
    /**
     * List all binary features corresponding to binary feature indexes in trees
     * @return
     */
    const TVector<TModelSplit>& GetBinFeatures() const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return RuntimeData->BinFeatures;
    }

    const TVector<TRepackedBin>& GetRepackedBins() const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return RuntimeData->RepackedBins;
    }

    const TVector<size_t>& GetFirstLeafOffsets() const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return RuntimeData->TreeFirstLeafOffsets;
    }

    const double* GetFirstLeafPtrForTree(size_t treeIdx) const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return &LeafValues[RuntimeData->TreeFirstLeafOffsets[treeIdx]];
    }

    /**
     * List all unique CTR bases (feature combination + ctr type) in model
     * @return
     */
    TVector<TModelCtrBase> GetUsedModelCtrBases() const {
        THashSet<TModelCtrBase> ctrsSet; // return sorted bases
        for (const auto& usedCtr : GetUsedModelCtrs()) {
            ctrsSet.insert(usedCtr.Base);
        }
        return TVector<TModelCtrBase>(ctrsSet.begin(), ctrsSet.end());
    }

    size_t GetNumFloatFeatures() const {
        if (FloatFeatures.empty()) {
            return 0;
        } else {
            return static_cast<size_t>(FloatFeatures.back().FeatureIndex) + 1;
        }
    }

    size_t GetMinimalSufficientFloatFeaturesVectorSize() const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return RuntimeData->MinimalSufficientFloatFeaturesVectorSize;
    }

    size_t GetUsedFloatFeaturesCount() const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return RuntimeData->UsedFloatFeaturesCount;
    }

    size_t GetNumCatFeatures() const {
        if (CatFeatures.empty()) {
            return 0;
        } else {
            return static_cast<size_t>(CatFeatures.back().FeatureIndex) + 1;
        }
    }

    size_t GetMinimalSufficientCatFeaturesVectorSize() const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return RuntimeData->MinimalSufficientCatFeaturesVectorSize;
    }

    size_t GetUsedCatFeaturesCount() const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return RuntimeData->UsedCatFeaturesCount;
    }

    size_t GetBinaryFeaturesFullCount() const {
        return GetBinFeatures().size();
    }

    ui32 GetEffectiveBinaryFeaturesBucketsCount() const {
        CB_ENSURE(RuntimeData.Defined(), "runtime data should be initialized");
        return RuntimeData->EffectiveBinFeaturesBucketCount;
    }

    size_t GetFlatFeatureVectorExpectedSize() const {
        return (size_t)Max(
            CatFeatures.empty() ? 0 : CatFeatures.back().FlatFeatureIndex + 1,
            FloatFeatures.empty() ? 0 : FloatFeatures.back().FlatFeatureIndex + 1
        );
    }

    TVector<ui32> GetTreeLeafCounts() const;

private:
    mutable TMaybe<TRuntimeData> RuntimeData;
};

/*!
 * \brief Full model class - contains all the data for model evaluation
 *
 * This class contains oblivious trees data, key-value dictionary for model metadata storage and CtrProvider
 *  holder.
 */
struct TFullModel {
    TObliviousTrees ObliviousTrees;
    /**
     * Model information key-value storage.
     */
    THashMap<TString, TString> ModelInfo;
    TIntrusivePtr<ICtrProvider> CtrProvider;

public:
    TFullModel() = default;

    bool operator==(const TFullModel& other) const {
        return ObliviousTrees == other.ObliviousTrees;
    }

    bool operator!=(const TFullModel& other) const {
        return !(*this == other);
    }

    void Swap(TFullModel& other) {
        DoSwap(ObliviousTrees, other.ObliviousTrees);
        DoSwap(ModelInfo, other.ModelInfo);
        DoSwap(CtrProvider, other.CtrProvider);
    }

    /**
     * Check whether model contains categorical features in OneHot conditions and/or CTR feature combinations
     */
    bool HasCategoricalFeatures() const {
        return GetUsedCatFeaturesCount() != 0;
    }

    /**
     * @return Number of trees in model.
     */
    size_t GetTreeCount() const {
        return ObliviousTrees.TreeSizes.size();
    }

    /**
     * @return Number of dimensions in model.
     */
    size_t GetDimensionsCount() const {
        return ObliviousTrees.ApproxDimension;
    }

    /**
     * Truncate trees to contain only trees from [begin; end) interval.
     * @param begin
     * @param end
     */
    void Truncate(size_t begin, size_t end) {
        ObliviousTrees.TruncateTrees(begin, end);
        if (CtrProvider) {
            CtrProvider->DropUnusedTables(ObliviousTrees.GetUsedModelCtrBases());
        }
        UpdateDynamicData();
    }

    /**
     * @return Minimal float features vector length sufficient for this model
     */
    size_t GetMinimalSufficientFloatFeaturesVectorSize() const {
        return ObliviousTrees.GetMinimalSufficientFloatFeaturesVectorSize();
    }
    /**
     * @return Number of float features that are really used in trees
     */
    size_t GetUsedFloatFeaturesCount() const {
        return ObliviousTrees.GetUsedFloatFeaturesCount();
    }

    /**
     * @return Expected float features vector length for this model
     */
    size_t GetNumFloatFeatures() const {
        return ObliviousTrees.GetNumFloatFeatures();
    }

    /**
     * @return Expected categorical features vector length for this model
     */
    size_t GetMinimalSufficientCatFeaturesVectorSize() const {
        return ObliviousTrees.GetMinimalSufficientCatFeaturesVectorSize();
    }
    /**
    * @return Number of float features that are really used in trees
    */
    size_t GetUsedCatFeaturesCount() const {
        return ObliviousTrees.GetUsedCatFeaturesCount();
    }

    /**
     * @return Expected categorical features vector length for this model
     */
    size_t GetNumCatFeatures() const {
        return ObliviousTrees.GetNumCatFeatures();
    }

    /**
     * Check whether model trees are oblivious
     */
    bool IsOblivious() const {
        return ObliviousTrees.IsOblivious();
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
        if (!CtrProvider) {
            return ObliviousTrees.GetUsedModelCtrs().empty();
        }
        return CtrProvider->HasNeededCtrs(ObliviousTrees.GetUsedModelCtrs());
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
        TArrayRef<double> results) const;

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
        TArrayRef<double> results) const;

    /**
     * Call CalcFlat on all model trees
     * @param features
     * @param results
     */
    void CalcFlat(TConstArrayRef<TConstArrayRef<float>> features, TArrayRef<double> results) const {
        CalcFlat(features, 0, ObliviousTrees.TreeSizes.size(), results);
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
        TArrayRef<double> results) const;

    /**
     * CalcFlatSingle on all trees in the model
     * @param[in] features flat features array reference. First dimension is object index, second dimension is
     *  feature index.
     * If feature is categorical, we do reinterpret cast from float to int.
     * @param[out] results double vector with indexation [classId].
     */
    void CalcFlatSingle(TConstArrayRef<float> features, TArrayRef<double> results) const {
        CalcFlatSingle(features, 0, ObliviousTrees.TreeSizes.size(), results);
    }

    /**
     * Shortcut for CalcFlatSingle
     */
    void CalcFlat(TConstArrayRef<float> features, TArrayRef<double> result) const {
        CalcFlatSingle(features, result);
    }

    /**
     * Staged model evaluation. Evaluates model for each incrementStep trees.
     * Useful for per tree model quality analysis.
     * @param[in] floatFeatures vector of float features values array references
     * @param[in] catFeatures vector of hashed categorical features values array references
     * @param[in] incrementStep tree count on each prediction stage
     * @return vector of vector of double - first index is for stage id, second is for
     *  [objectIndex * ApproxDimension + classId]
     */
    TVector<TVector<double>> CalcTreeIntervals(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TConstArrayRef<int>> catFeatures,
        size_t incrementStep) const;

    /**
     * Same as CalcTreeIntervalsFlat but for **flat** feature vectors
     * @param[in] mixedFeatures
     * @param[in] incrementStep
     * @return
     */
    TVector<TVector<double>> CalcTreeIntervalsFlat(
        TConstArrayRef<TConstArrayRef<float>> mixedFeatures,
        size_t incrementStep) const;

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
        TArrayRef<double> results) const;

    /**
     * Evaluate raw formula predictions on user data. Uses all model trees
     * @param floatFeatures
     * @param catFeatures hashed cat feature values
     * @param results results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TConstArrayRef<int>> catFeatures,
        TArrayRef<double> results) const {

        Calc(floatFeatures, catFeatures, 0, ObliviousTrees.TreeSizes.size(), results);
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
        TArrayRef<double> result) const {

        const TConstArrayRef<float> floatFeaturesArray[] = {floatFeatures};
        const TConstArrayRef<int> catFeaturesArray[] = {catFeatures};
        Calc(floatFeaturesArray, catFeaturesArray, result);
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
        TArrayRef<double> results) const;

    /**
     * Evaluate raw formula predictions for objects. Uses all model trees.
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @param results indexation is [objectIndex * ApproxDimension + classId]
     */
    void Calc(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TStringBuf>> catFeatures,
        TArrayRef<double> results) const {

        Calc(floatFeatures, catFeatures, 0, ObliviousTrees.TreeSizes.size(), results);
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
        TArrayRef<ui32> indexes) const;

    /**
     * Evaluate indexes of leafs at which object are mapped by all trees of the model.
     * @param floatFeatures
     * @param catFeatures vector of TStringBuf with categorical features strings
     * @return indexes; size should be equal to number of trees in the model.
     */
    void CalcLeafIndexesSingle(
        TConstArrayRef<float> floatFeatures,
        TConstArrayRef<TStringBuf> catFeatures,
        TArrayRef<ui32> indexes
    ) const {
        CalcLeafIndexesSingle(floatFeatures, catFeatures, 0, ObliviousTrees.TreeSizes.size(), indexes);
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
        TConstArrayRef<TVector<TStringBuf>> catFeatures,
        size_t treeStart,
        size_t treeEnd,
        TArrayRef<ui32> indexes
    ) const;

    /**
     * Evaluate indexes of leafs at which objects are mapped by all trees of the model.
     * @param floatFeatures
     * @param catFeatures vector of vector of TStringBuf with categorical features strings
     * @return indexes; indexation is [objectIndex * treeCount + treeIndex]
     */
    void CalcLeafIndexes(
        TConstArrayRef<TConstArrayRef<float>> floatFeatures,
        TConstArrayRef<TVector<TStringBuf>> catFeatures,
        TArrayRef<ui32> indexes
    ) const {
        CalcLeafIndexes(floatFeatures, catFeatures, 0, ObliviousTrees.TreeSizes.size(), indexes);
    }

    /**
     * Get the name of optimized objective used to train the model.
     * @return the name, or empty string if the model does not have this information
     */
    TString GetLossFunctionName() const;

    /**
     * Internal usage only.
     * Updates indexes in CTR provider and recalculates runtime data in Oblivious trees after model
     *  modifications.
     */
    void UpdateDynamicData() {
        ObliviousTrees.UpdateRuntimeData();
        if (CtrProvider) {
            CtrProvider->SetupBinFeatureIndexes(
                ObliviousTrees.FloatFeatures,
                ObliviousTrees.OneHotFeatures,
                ObliviousTrees.CatFeatures);
        }
    }
};

void OutputModel(const TFullModel& model, TStringBuf modelFile);
void OutputModel(const TFullModel& model, IOutputStream* out);
TFullModel ReadModel(const TString& modelFile, EModelType format = EModelType::CatboostBinary);
TFullModel ReadModel(
    const void* binaryBuffer,
    size_t binaryBufferSize,
    EModelType format = EModelType::CatboostBinary);

/**
 * Export model in our binary or protobuf CoreML format
 * @param model
 * @param modelFile
 * @param format
 * @param userParametersJson
 * @param addFileFormatExtension
 * @param featureId
 * @param catFeaturesHashToString
 */
void ExportModel(
    const TFullModel& model,
    const TString& modelFile,
    EModelType format,
    const TString& userParametersJson = "",
    bool addFileFormatExtension = false,
    const TVector<TString>* featureId=nullptr,
    const THashMap<ui32, TString>* catFeaturesHashToString=nullptr);

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

//TODO(kirillovs): make this method a member of TFullModel
TVector<TString> GetModelClassNames(const TFullModel& model);

TFullModel SumModels(
    const TVector<const TFullModel*> modelVector,
    const TVector<double>& weights,
    ECtrTableMergePolicy ctrMergePolicy = ECtrTableMergePolicy::IntersectingCountersAverage);

void SaveModelBorders(
    const TString& file,
    const TFullModel& model);
