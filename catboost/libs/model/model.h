#pragma once

#include "features.h"
#include "split.h"

#include "static_ctr_provider.h"

#include <catboost/libs/model/flatbuffers/model.fbs.h>

#include <catboost/libs/cat_feature/cat_feature.h>

#include <library/json/json_reader.h>

#include <util/system/mutex.h>
#include <util/stream/file.h>

class TModelPartsCachingSerializer;

/*!
    \brief Oblivious tree model structure

    This structure contains the data about tree conditions and leaf values.
    We use oblivious trees - symmetric trees that has the same binary condition on each level.
    So each leaf index is determined by binary vector with length equal to evaluated tree depth.

    That allows us to evaluate model predictions very fast (even without planned SIMD optimizations) compared to asymmetric trees.

    Our oblivious tree model can contain float, one-hot and CTR binary conditions:
    - Float condition - float feature value is greater than float border
    - One-hot condition - hashed cat feature value is equal to some value
    - CTR condition - calculated ctr is greater than float border
    You can read about CTR calculation in ctr_provider.h
*/

struct TObliviousTrees {
    struct TMetaData { //contains data calculated in runtime on model load/update
        TVector<TModelCtr> UsedModelCtrs;
        TVector<TModelSplit> BinFeatures;
    };
    int ApproxDimension = 1;
    // Tree splits stored in 3 vectors<int>:
    // Split values
    TVector<int> TreeSplits;
    // Tree sizes
    TVector<int> TreeSizes;
    // Offset of first split in TreeSplits array
    TVector<int> TreeStartOffsets;

    //Leaf values layout: [treeIndex][leafId * ApproxDimension + dimension]
    TVector<TVector<double>> LeafValues;

    //Cat features, used in model
    TVector<TCatFeature> CatFeatures;

    static_assert(ESplitType::FloatFeature < ESplitType::OneHotFeature
                  && ESplitType::OneHotFeature < ESplitType::OnlineCtr,
                  "ESplitType should represent bin feature order in model");

    // Those 3 vectors are form binary features sequence
    // Float features used in model
    TVector<TFloatFeature> FloatFeatures;
    // One hot encoded features used in model
    TVector<TOneHotFeature> OneHotFeatures;
    // CTR features used in model
    TVector<TCtrFeature> CtrFeatures;

    flatbuffers::Offset<NCatBoostFbs::TObliviousTrees> FBSerialize(TModelPartsCachingSerializer& serializer) const;
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
        LeafValues.resize(TreeSizes.size());
        if (fbObj->LeafValues()) {
            auto leafValIter = fbObj->LeafValues()->begin();
            for (size_t treeId = 0; treeId < TreeSizes.size(); ++treeId) {
                const auto treeLeafCout = ApproxDimension * (1 << TreeSizes[treeId]);
                LeafValues[treeId].assign(leafValIter, leafValIter + treeLeafCout);
                leafValIter += treeLeafCout;
            }
        }

#define FEATURES_ARRAY_DESERIALIZER(var) \
        if (fbObj->var()) {\
            var.resize(fbObj->var()->size());\
            for (size_t i = 0; i < fbObj->var()->size(); ++i) {\
                var[i].FBDeserialize(fbObj->var()->Get(i));\
            }\
        }
        FEATURES_ARRAY_DESERIALIZER(CatFeatures)
        FEATURES_ARRAY_DESERIALIZER(FloatFeatures)
        FEATURES_ARRAY_DESERIALIZER(OneHotFeatures)
        FEATURES_ARRAY_DESERIALIZER(CtrFeatures)
#undef FEATURES_ARRAY_DESERIALIZER
    }

    void AddBinTree(const TVector<int>& binSplits) {
        Y_ASSERT(TreeSplits.size() == TreeSizes.size() && TreeSizes.size() == TreeStartOffsets.size());
        TreeSplits.insert(TreeSplits.end(), binSplits.begin(), binSplits.end());
        TreeSizes.push_back(binSplits.ysize());
        if (TreeStartOffsets.empty()) {
            TreeStartOffsets.push_back(0);
        } else {
            TreeStartOffsets.push_back(TreeStartOffsets.back() + binSplits.ysize());
        }
    }

    bool operator==(const TObliviousTrees& other) const {
        return std::tie(ApproxDimension,
                        TreeSplits,
                        TreeSizes,
                        TreeStartOffsets,
                        LeafValues,
                        CatFeatures,
                        FloatFeatures,
                        OneHotFeatures,
                        CtrFeatures)
           == std::tie(other.ApproxDimension,
                       other.TreeSplits,
                       other.TreeSizes,
                       other.TreeStartOffsets,
                       other.LeafValues,
                       other.CatFeatures,
                       other.FloatFeatures,
                       other.OneHotFeatures,
                       other.CtrFeatures);
    }
    bool operator!=(const TObliviousTrees& other) const {
        return !(*this == other);
    }
    size_t GetTreeCount() const {
        return TreeSizes.size();
    }

    void Truncate(size_t begin, size_t end);

    void UpdateMetadata() const {
        MetaData = TMetaData{}; // reset metadata
        auto& ref = MetaData.GetRef();
        for (const auto& ctrFeature : CtrFeatures) {
            ref.UsedModelCtrs.push_back(ctrFeature.Ctr);
        }

        for (const auto& feature: FloatFeatures) {
            for (int borderId = 0; borderId < feature.Borders.ysize(); ++borderId) {
                TFloatSplit fs{feature.FeatureIndex, feature.Borders[borderId]};
                ref.BinFeatures.emplace_back(fs);
            }
        }
        for (const auto& feature: OneHotFeatures) {
            for (int valueId = 0; valueId < feature.Values.ysize(); ++valueId) {
                TOneHotSplit oh{feature.CatFeatureIndex, feature.Values[valueId]};
                ref.BinFeatures.emplace_back(oh);
            }
        }
        for (const auto& feature: CtrFeatures) {
            for (int borderId = 0; borderId < feature.Borders.ysize(); ++borderId) {
                TModelCtrSplit ctrSplit;
                ctrSplit.Ctr = feature.Ctr;
                ctrSplit.Border = feature.Borders[borderId];
                ref.BinFeatures.emplace_back(std::move(ctrSplit));
            }
        }
    }

    const TVector<TModelCtr>& GetUsedModelCtrs() const {
        if (MetaData.Defined()) {
            return MetaData->UsedModelCtrs;
        }
        UpdateMetadata();

        return MetaData->UsedModelCtrs;
    }

    const TVector<TModelSplit>& GetBinFeatures() const {
        if (MetaData.Defined()) {
            return MetaData->BinFeatures;
        }
        UpdateMetadata();

        return MetaData->BinFeatures;
    }

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
            return static_cast<size_t>(FloatFeatures.back().FeatureIndex + 1);
        }
    }

    size_t GetNumCatFeatures() const {
        if (CatFeatures.empty()) {
            return 0;
        } else {
            return static_cast<size_t>(CatFeatures.back().FeatureIndex + 1);
        }
    }

    size_t GetBinaryFeaturesCount() const {
        return GetBinFeatures().size();
    }

    size_t GetFlatFeatureVectorExpectedSize() const {
        return GetNumFloatFeatures() + GetNumCatFeatures();
    }
private:
    mutable TMaybe<TMetaData> MetaData;
};

struct TFullModel {
    TObliviousTrees ObliviousTrees;

    THashMap<TString, TString> ModelInfo;
    TIntrusivePtr<ICtrProvider> CtrProvider;

    void Swap(TFullModel& other) {
        DoSwap(ObliviousTrees, other.ObliviousTrees);
        DoSwap(ModelInfo, other.ModelInfo);
        DoSwap(CtrProvider, other.CtrProvider);
    }

    bool HasCategoricalFeatures() const {
        return !ObliviousTrees.CatFeatures.empty();
    }

    size_t GetTreeCount() const {
        return ObliviousTrees.TreeSizes.size();
    }

    size_t GetNumFloatFeatures() const {
        return ObliviousTrees.GetNumFloatFeatures();
    }

    size_t GetNumCatFeatures() const {
        return ObliviousTrees.GetNumCatFeatures();
    }

    TFullModel() = default;

    bool operator==(const TFullModel& other) const {
        return std::tie(ObliviousTrees, ModelInfo) ==
               std::tie(other.ObliviousTrees, other.ModelInfo);
    }

    bool operator!=(const TFullModel& other) const {
        return !(*this == other);
    }

    void Save(IOutputStream* s) const;
    void Load(IInputStream* s);

    // if no ctr features present it'll return false
    bool HasValidCtrProvider() const {
        if (!CtrProvider) {
            return false;
        }
        return CtrProvider->HasNeededCtrs(ObliviousTrees.GetUsedModelCtrs());
    }

    void CalcFlatTransposed(const TVector<TConstArrayRef<float>>& transposedFeatures, size_t treeStart, size_t treeEnd, TArrayRef<double> results) const;
    void CalcFlat(const TVector<TConstArrayRef<float>>& features, size_t treeStart, size_t treeEnd, TArrayRef<double> results) const;
    void CalcFlatSingle(const TConstArrayRef<float>& features, size_t treeStart, size_t treeEnd, TArrayRef<double> results) const;
    void CalcFlatSingle(const TConstArrayRef<float>& features, TArrayRef<double> results) const {
        CalcFlatSingle(features, 0, ObliviousTrees.TreeSizes.size(), results);
    }
    void CalcFlat(const TVector<TConstArrayRef<float>>& features, TArrayRef<double> results) const {
        CalcFlat(features, 0, ObliviousTrees.TreeSizes.size(), results);
    }
    void CalcFlat(TConstArrayRef<float> features, TArrayRef<double> result) const {
        TVector<TConstArrayRef<float>> featuresVec = {features};
        CalcFlat(featuresVec, result);
    }

    TVector<TVector<double>> CalcTreeIntervals(
        const TVector<TConstArrayRef<float>>& floatFeatures,
        const TVector<TConstArrayRef<int>>& catFeatures,
        size_t incrementStep) const;

    TVector<TVector<double>> CalcTreeIntervalsFlat(
        const TVector<TConstArrayRef<float>>& mixedFeatures,
        size_t incrementStep) const;

    void Calc(const TVector<TConstArrayRef<float>>& floatFeatures,
              const TVector<TConstArrayRef<int>>& catFeatures,
              size_t treeStart,
              size_t treeEnd,
              TArrayRef<double> results) const;
    void Calc(const TVector<TConstArrayRef<float>>& floatFeatures,
              const TVector<TConstArrayRef<int>>& catFeatures,
              TArrayRef<double> results) const {
        Calc(floatFeatures, catFeatures, 0, ObliviousTrees.TreeSizes.size(), results);
    }

    void Calc(TConstArrayRef<float> floatFeatures,
              TConstArrayRef<int> catFeatures,
              TArrayRef<double> result) const {
        TVector<TConstArrayRef<float>> floatFeaturesVec = {floatFeatures};
        TVector<TConstArrayRef<int>> catFeaturesVec = {catFeatures};
        Calc(floatFeaturesVec, catFeaturesVec, result);
    }

    void Calc(const TVector<TConstArrayRef<float>>& floatFeatures,
              const TVector<TVector<TStringBuf>>& catFeatures,
              size_t treeStart,
              size_t treeEnd,
              TArrayRef<double> results) const;

    void Calc(const TVector<TConstArrayRef<float>>& floatFeatures,
              const TVector<TVector<TStringBuf>>& catFeatures,
              TArrayRef<double> results) const {
        Calc(floatFeatures, catFeatures, 0, ObliviousTrees.TreeSizes.size(), results);
    }

    TFullModel CopyTreeRange(size_t begin, size_t end) const {
        TFullModel result = *this;
        result.ObliviousTrees.Truncate(begin, end);
        return result;
    }
    void UpdateDynamicData() {
        ObliviousTrees.UpdateMetadata();
        if (CtrProvider) {
            CtrProvider->SetupBinFeatureIndexes(
                ObliviousTrees.FloatFeatures,
                ObliviousTrees.OneHotFeatures,
                ObliviousTrees.CatFeatures);
        }
    }
};

void OutputModel(const TFullModel& model, const TString& modelFile);
TFullModel ReadModel(const TString& modelFile);

enum class EModelExportType {
    CatboostBinary,
    AppleCoreML
};

void ExportModel(const TFullModel& model, const TString& modelFile, const EModelExportType format = EModelExportType::CatboostBinary, const TString& userParametersJSON = "");

TString SerializeModel(const TFullModel& model);
TFullModel DeserializeModel(const TString& serializeModelString);
