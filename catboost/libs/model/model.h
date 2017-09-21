#pragma once

#include "target_classifier.h"

#include <catboost/libs/model/online_ctr.h>
#include <catboost/libs/model/tensor_struct.h>

#include <library/json/json_reader.h>

#include <util/generic/vector.h>
#include <util/system/mutex.h>
#include <util/stream/file.h>

class TCtrValueTable {
public:
    TDenseHash<ui64, ui32> Hash;

    int CounterDenominator = 0;
    int TargetClassesCount = 0;


    template<typename T>
    NArrayRef::TConstArrayRef<T> GetTypedArrayRefForBlobData() const {
        return MakeArrayRef(
            reinterpret_cast<const T*>(CTRBlob.data()),
            CTRBlob.size() / sizeof(T)
        );
    }

    template<typename T>
    NArrayRef::TArrayRef<T> AllocateBlobAndGetArrayRef(size_t elementCount) {
        CTRBlob.resize(elementCount * sizeof(T));
        std::fill(CTRBlob.begin(), CTRBlob.end(), 0);
        return MakeArrayRef(
            reinterpret_cast<T*>(CTRBlob.data()),
            elementCount
        );
    }

    bool operator==(const TCtrValueTable& other) const {
        return std::tie(Hash, CTRBlob, CounterDenominator, TargetClassesCount) ==
               std::tie(other.Hash, other.CTRBlob, other.CounterDenominator, other.TargetClassesCount);
    }

    static const ui64 UnknownHash = (ui64)-1;

    inline ui64 ResolveHashToIndex(ui64 hash) const {
        auto val = Hash.FindPtr(hash);
        if (val) {
            return (ui64)*val;
        }
        return UnknownHash;
    }

    Y_SAVELOAD_DEFINE(Hash, CTRBlob, CounterDenominator, TargetClassesCount)

private:
    yvector<ui8> CTRBlob;
};

struct TCtrData {
    using TLearnCtrHash = yhash<TModelCtrBase, TCtrValueTable>;
    TLearnCtrHash LearnCtrs;

    bool operator==(const TCtrData& other) const {
        return LearnCtrs == other.LearnCtrs;
    }

    bool operator!=(const TCtrData& other) const {
        return !(*this == other);
    }

    inline void Save(IOutputStream* s) const {
        // all this looks like yserializer for hash copypaste implementation
        // but we have to do this to allow partially streamed serialization of big models
        ::SaveSize(s, LearnCtrs.size());
        ::SaveRange(s, LearnCtrs.begin(), LearnCtrs.end());
    }

    inline void Load(IInputStream* s) {
        const size_t cnt = ::LoadSize(s);
        LearnCtrs.reserve(cnt);

        for (size_t i = 0; i != cnt; ++i) {
            std::pair<TModelCtrBase, TCtrValueTable> kv;
            ::Load(s, kv);
            LearnCtrs.emplace(std::move(kv));
        }
    }
};

struct TOneHotFeaturesInfo {
    yhash<int, TString> FeatureHashToOrigString;
    bool operator==(const TOneHotFeaturesInfo& other) const {
        return FeatureHashToOrigString == other.FeatureHashToOrigString;
    }

    bool operator!=(const TOneHotFeaturesInfo& other) const {
        return !(*this == other);
    }
    Y_SAVELOAD_DEFINE(FeatureHashToOrigString);
};

struct TCoreModel {
    yvector<bool> HasNans;
    yvector<yvector<float>> Borders;
    yvector<TTensorStructure3> TreeStruct;
    yvector<yvector<yvector<double>>> LeafValues; // [numTree][dim][bucketId]
    yvector<int> CatFeatures;
    yvector<TString> FeatureIds;
    int FeatureCount = 0;
    int ApproxDimension = 1;
    yvector<TTargetClassifier> TargetClassifiers;
    yhash<TString, TString> ModelInfo;

    bool operator==(const TCoreModel& other) const {
        return std::tie(HasNans, Borders, TreeStruct, LeafValues, CatFeatures, FeatureIds, FeatureCount, ApproxDimension, TargetClassifiers, ModelInfo) ==
               std::tie(other.HasNans, other.Borders, other.TreeStruct, other.LeafValues, other.CatFeatures, other.FeatureIds, other.FeatureCount, other.ApproxDimension, other.TargetClassifiers, other.ModelInfo);
    }

    bool operator!=(const TCoreModel& other) const {
        return !(*this == other);
    }

    void Swap(TCoreModel& other) {
        DoSwap(HasNans, other.HasNans);
        DoSwap(Borders, other.Borders);
        DoSwap(TreeStruct, other.TreeStruct);
        DoSwap(LeafValues, other.LeafValues);
        DoSwap(CatFeatures, other.CatFeatures);
        DoSwap(FeatureIds, other.FeatureIds);
        DoSwap(FeatureCount, other.FeatureCount);
        DoSwap(ApproxDimension, other.ApproxDimension);
        DoSwap(TargetClassifiers, other.TargetClassifiers);
        DoSwap(ModelInfo, other.ModelInfo);
    }

    Y_SAVELOAD_DEFINE(HasNans, Borders, TreeStruct, LeafValues, CatFeatures, FeatureIds, FeatureCount, ApproxDimension, TargetClassifiers, ModelInfo)
};

struct TFullModel: public TCoreModel {
    TOneHotFeaturesInfo OneHotFeaturesInfo;
    TCtrData CtrCalcerData;

    TFullModel& operator=(TCoreModel&& model) {
        TCoreModel::operator=(std::move(model));
        return *this;
    }

    bool operator==(const TFullModel& other) const {
        return TCoreModel::operator==(other) && CtrCalcerData == other.CtrCalcerData;
    }

    void Save(IOutputStream* s) const {
        TCoreModel::Save(s);
        ::Save(s, OneHotFeaturesInfo);
        ::Save(s, CtrCalcerData);
    }
    void Load(IInputStream* s) {
        TCoreModel::Load(s);
        ::Load(s, OneHotFeaturesInfo);
        ::Load(s, CtrCalcerData);
    }

    void Swap(TFullModel& other) {
        TCoreModel::Swap(other);
        DoSwap(CtrCalcerData, other.CtrCalcerData);
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

class TStreamedFullModelSaver {
public:
    TStreamedFullModelSaver(const TString& filename, size_t ctrValuesCount, const TCoreModel& coreModel, const TOneHotFeaturesInfo& oneHotFeaturesInfo)
        : Stream(filename)
        , ExpectedWritesCount(ctrValuesCount)
    {
        ::Save(&Stream, coreModel);
        ::Save(&Stream, oneHotFeaturesInfo);
        ::SaveSize(&Stream, ExpectedWritesCount);
    }

    void SaveOneCtr(const TModelCtrBase& ctr, const TCtrValueTable& valTable) {
        with_lock (StreamLock) {
            ++WritesCount;
            ::SaveMany(&Stream, ctr, valTable);
        }
    }

    ~TStreamedFullModelSaver() {
        Y_VERIFY(WritesCount == ExpectedWritesCount);
    }

private:
    TMutex StreamLock;
    TOFStream Stream;
    size_t WritesCount = 0;
    size_t ExpectedWritesCount = 0;
};
