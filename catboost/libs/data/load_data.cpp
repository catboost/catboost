#include "load_data.h"
#include "doc_pool_data_provider.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/options/restrictions.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>

namespace NCB {

    namespace {

    class TPoolBuilder: public IPoolBuilder {
    public:
        TPoolBuilder(const NPar::TLocalExecutor& localExecutor, TPool* pool)
            : Pool(pool)
            , LocalExecutor(localExecutor)
        {
        }

        void Start(const TPoolMetaInfo& poolMetaInfo,
                   int docCount,
                   const TVector<int>& catFeatureIds) override {
            Cursor = NotSet;
            NextCursor = 0;
            FeatureCount = poolMetaInfo.FeatureCount;
            BaselineCount = poolMetaInfo.BaselineCount;
            Pool->Docs.Resize(docCount,
                              FeatureCount,
                              BaselineCount,
                              poolMetaInfo.HasGroupId,
                              poolMetaInfo.HasSubgroupIds);
            Pool->CatFeatures = catFeatureIds;
            Pool->FeatureId.assign(FeatureCount, TString());
            Pool->MetaInfo = poolMetaInfo;
        }

        void StartNextBlock(ui32 blockSize) override {
            Cursor = NextCursor;
            NextCursor = Cursor + blockSize;
        }

        float GetCatFeatureValue(const TStringBuf& feature) override {
            int hashVal = CalcCatFeatureHash(feature);
            int hashPartIdx = LocalExecutor.GetWorkerThreadId();
            CB_ENSURE(hashPartIdx < CB_THREAD_LIMIT, "Internal error: thread ID exceeds CB_THREAD_LIMIT");
            auto& curPart = HashMapParts[hashPartIdx];
            if (!curPart.CatFeatureHashes.has(hashVal)) {
                curPart.CatFeatureHashes[hashVal] = feature;
            }
            return ConvertCatFeatureHashToFloat(hashVal);
        }

        void AddCatFeature(ui32 localIdx, ui32 featureId, const TStringBuf& feature) override {
            AddFloatFeature(localIdx, featureId, GetCatFeatureValue(feature));
        }

        void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) override {
            Pool->Docs.Factors[featureId][Cursor + localIdx] = feature;
        }

        void AddBinarizedFloatFeature(ui32 localIdx, ui32 featureId, ui8 binarizedFeature) override {
            Y_UNUSED(localIdx);
            Y_UNUSED(featureId);
            Y_UNUSED(binarizedFeature);
            CB_ENSURE(false, "Not supported for regular pools");
        }

        void AddAllFloatFeatures(ui32 localIdx, TConstArrayRef<float> features) override {
            CB_ENSURE(features.size() == FeatureCount, "Error: number of features should be equal to factor count");
            TVector<float>* factors = Pool->Docs.Factors.data();
            for (ui32 featureId = 0; featureId < FeatureCount; ++featureId) {
                factors[featureId][Cursor + localIdx] = features[featureId];
            }
        }

        void AddLabel(ui32 localIdx, const TStringBuf& label) override {
            Pool->Docs.Label[Cursor + localIdx] = label;
        }

        void AddTarget(ui32 localIdx, float value) override {
            Pool->Docs.Target[Cursor + localIdx] = value;
        }

        void AddWeight(ui32 localIdx, float value) override {
            Pool->Docs.Weight[Cursor + localIdx] = value;
        }

        void AddQueryId(ui32 localIdx, TGroupId value) override {
            Pool->Docs.QueryId[Cursor + localIdx] = value;
        }

        void AddBaseline(ui32 localIdx, ui32 offset, double value) override {
            Pool->Docs.Baseline[offset][Cursor + localIdx] = value;
        }

        void AddDocId(ui32 localIdx, const TStringBuf& value) override {
            Pool->Docs.Id[Cursor + localIdx] = value;
        }

        void AddSubgroupId(ui32 localIdx, TSubgroupId value) override {
            Pool->Docs.SubgroupId[Cursor + localIdx] = value;
        }

        void AddTimestamp(ui32 localIdx, ui64 value) override {
            Pool->Docs.Timestamp[Cursor + localIdx] = value;
        }

        void SetFeatureIds(const TVector<TString>& featureIds) override {
            Y_ENSURE(featureIds.size() == FeatureCount, "Error: feature ids size should be equal to factor count");
            Pool->FeatureId = featureIds;
        }

        void SetPairs(const TVector<TPair>& pairs) override {
            Pool->Pairs = pairs;
        }

        void SetGroupWeights(const TVector<float>& groupWeights) override {
            CB_ENSURE(Pool->Docs.GetDocCount() == groupWeights.size(),
                "Group weights file should have as many weights as the objects in the dataset.");
            Pool->Docs.Weight = groupWeights;
        }

        void SetTarget(const TVector<float>& target) override {
            Pool->Docs.Target = target;
        }

        void SetFloatFeatures(const TVector<TFloatFeature>& floatFeatures) override {
            Y_UNUSED(floatFeatures);
            CB_ENSURE(false, "Not supported for regular pools");
        }

        int GetDocCount() const override {
            return NextCursor;
        }

        TConstArrayRef<TString> GetLabels() const override {
            return MakeArrayRef(Pool->Docs.Label.data(), Pool->Docs.Label.size());
        }

        TConstArrayRef<float> GetWeight() const override {
            return MakeArrayRef(Pool->Docs.Weight.data(), Pool->Docs.Weight.size());
        }

        TConstArrayRef<TGroupId> GetGroupIds() const override {
            return MakeArrayRef(Pool->Docs.QueryId.data(), Pool->Docs.QueryId.size());
        }

        void GenerateDocIds(int offset) override {
            for (int ind = 0; ind < Pool->Docs.Id.ysize(); ++ind) {
                Pool->Docs.Id[ind] = ToString(offset + ind);
            }
        }

        void Finish() override {
            if (Pool->Docs.GetDocCount() != 0) {
                for (const auto& part : HashMapParts) {
                    Pool->CatFeaturesHashToString.insert(part.CatFeatureHashes.begin(), part.CatFeatureHashes.end());
                }
                MATRIXNET_INFO_LOG << "Doc info sizes: " << Pool->Docs.GetDocCount() << " " << FeatureCount << Endl;
            } else {
                MATRIXNET_ERROR_LOG << "No doc info loaded" << Endl;
            }
        }

    private:
        struct THashPart {
            THashMap<int, TString> CatFeatureHashes;
        };
        TPool* Pool;
        static constexpr const int NotSet = -1;
        ui32 Cursor = NotSet;
        ui32 NextCursor = 0;
        ui32 FeatureCount = 0;
        ui32 BaselineCount = 0;
        std::array<THashPart, CB_THREAD_LIMIT> HashMapParts;
        const NPar::TLocalExecutor& LocalExecutor;
    };


    class TQuantizedBuilder: public IPoolBuilder { // TODO(akhropov): Temporary solution until MLTOOLS-140 is implemented
    public:
        TQuantizedBuilder(TPool* pool)
            : Pool(pool)
        {
        }

        void Start(const TPoolMetaInfo& poolMetaInfo,
                   int docCount,
                   const TVector<int>& catFeatureIds) override {
            Cursor = NotSet;
            NextCursor = 0;
            FeatureCount = poolMetaInfo.FeatureCount;
            BaselineCount = poolMetaInfo.BaselineCount;
            ResizePool(docCount, poolMetaInfo);
            Pool->CatFeatures = catFeatureIds;
        }

        void StartNextBlock(ui32 blockSize) override {
            Cursor = NextCursor;
            NextCursor = Cursor + blockSize;
        }

        float GetCatFeatureValue(const TStringBuf& feature) override {
            Y_UNUSED(feature);
            CB_ENSURE(false, "Not supported for binarized pools");
        }

        void AddCatFeature(ui32 localIdx, ui32 featureId, const TStringBuf& feature) override {
            AddFloatFeature(localIdx, featureId, GetCatFeatureValue(feature));
        }

        void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) override {
            Y_UNUSED(localIdx);
            Y_UNUSED(featureId);
            Y_UNUSED(feature);
            CB_ENSURE(false, "Not supported for binarized pools");
        }

        void AddBinarizedFloatFeature(ui32 localIdx, ui32 featureId, ui8 binarizedFeature) override {
            if (Pool->QuantizedFeatures.FloatHistograms[featureId].empty()) {
                Pool->QuantizedFeatures.FloatHistograms[featureId].resize(Pool->Docs.GetDocCount());
            }
            Pool->QuantizedFeatures.FloatHistograms[featureId][Cursor + localIdx] = binarizedFeature;
        }

        void AddAllFloatFeatures(ui32 localIdx, TConstArrayRef<float> features) override {
            Y_UNUSED(localIdx);
            Y_UNUSED(features);
            CB_ENSURE(false, "Not supported for binarized pools");
        }

        void AddLabel(ui32 localIdx, const TStringBuf& label) override {
            Pool->Docs.Label[Cursor + localIdx] = label;
        }

        void AddTarget(ui32 localIdx, float value) override {
            Pool->Docs.Target[Cursor + localIdx] = value;
        }

        void AddWeight(ui32 localIdx, float value) override {
            Pool->Docs.Weight[Cursor + localIdx] = value;
        }

        void AddQueryId(ui32 localIdx, TGroupId value) override {
            Pool->Docs.QueryId[Cursor + localIdx] = value;
        }

        void AddBaseline(ui32 localIdx, ui32 offset, double value) override {
            Pool->Docs.Baseline[offset][Cursor + localIdx] = value;
        }

        void AddDocId(ui32 localIdx, const TStringBuf& value) override {
            Y_UNUSED(localIdx);
            Y_UNUSED(value);
            CB_ENSURE(false, "Not supported for binarized pools");
        }

        void AddSubgroupId(ui32 localIdx, TSubgroupId value) override {
            Pool->Docs.SubgroupId[Cursor + localIdx] = value;
        }

        void AddTimestamp(ui32 localIdx, ui64 value) override {
            Y_UNUSED(localIdx);
            Y_UNUSED(value);
            CB_ENSURE(false, "Not supported for binarized pools");
        }

        void SetFeatureIds(const TVector<TString>& featureIds) override {
            CB_ENSURE(featureIds.size() == FeatureCount, "Error: feature ids size should be equal to factor count");
        }

        void SetPairs(const TVector<TPair>& pairs) override {
            Pool->Pairs = pairs;
        }

        void SetGroupWeights(const TVector<float>& groupWeights) override {
            CB_ENSURE(Pool->Docs.GetDocCount() == groupWeights.size(),
                "Group weights file should have as many weights as the objects in the dataset.");
            Pool->Docs.Weight = groupWeights;
        }

        void SetTarget(const TVector<float>& target) override {
            Pool->Docs.Target = target;
        }

        void SetFloatFeatures(const TVector<TFloatFeature>& floatFeatures) override {
            Pool->FloatFeatures = floatFeatures;
        }

        int GetDocCount() const override {
            return NextCursor;
        }

        TConstArrayRef<TString> GetLabels() const override {
            return MakeArrayRef(Pool->Docs.Label.data(), Pool->Docs.Label.size());
        }

        TConstArrayRef<float> GetWeight() const override {
            return MakeArrayRef(Pool->Docs.Weight.data(), Pool->Docs.Weight.size());
        }

        TConstArrayRef<TGroupId> GetGroupIds() const override {
            return MakeArrayRef(Pool->Docs.QueryId.data(), Pool->Docs.QueryId.size());
        }

        void GenerateDocIds(int offset) override {
            for (int ind = 0; ind < Pool->Docs.Id.ysize(); ++ind) {
                Pool->Docs.Id[ind] = ToString(offset + ind);
            }
        }

        void Finish() override {
            if (Pool->QuantizedFeatures.GetDocCount() != 0) {
                MATRIXNET_INFO_LOG << "Doc info sizes: " << Pool->QuantizedFeatures.GetDocCount() << " " << FeatureCount << Endl;
            } else {
                MATRIXNET_ERROR_LOG << "No doc info loaded" << Endl;
            }
        }

    private:
        void ResizePool(int docCount, const TPoolMetaInfo& metaInfo) {
            // setup numerical features
            CB_ENSURE(metaInfo.ColumnsInfo.Defined(), "Missing column info");
            Pool->QuantizedFeatures.FloatHistograms.resize(metaInfo.ColumnsInfo->CountColumns(EColumn::Num));
            // setup cat features
            // TODO(yazevnul): support cat features in quantized pools
            const ui32 catFeaturesCount = metaInfo.ColumnsInfo->CountColumns(EColumn::Categ);
            Pool->QuantizedFeatures.CatFeaturesRemapped.resize(catFeaturesCount);
            Pool->QuantizedFeatures.OneHotValues.resize(catFeaturesCount, TVector<int>(/*one hot dummy size*/ 1, /*dummy value*/ 0));
            Pool->QuantizedFeatures.IsOneHot.resize(catFeaturesCount, /*isOneHot*/ false);
            // setup rest
            Pool->Docs.Baseline.resize(metaInfo.BaselineCount);
            for (auto& dim : Pool->Docs.Baseline) {
                dim.resize(docCount);
            }
            Pool->Docs.Target.resize(docCount);
            Pool->Docs.Weight.resize(docCount, 1.0f);
            if (metaInfo.HasGroupId) {
                Pool->Docs.QueryId.resize(docCount);
            }
            if (metaInfo.HasSubgroupIds) {
                Pool->Docs.SubgroupId.resize(docCount);
            }
            Pool->Docs.Timestamp.resize(docCount);
        }

        TPool* Pool;
        static constexpr const int NotSet = -1;
        ui32 Cursor = NotSet;
        ui32 NextCursor = 0;
        ui32 FeatureCount = 0;
        ui32 BaselineCount = 0;
    };
    } // anonymous namespace

    TTargetConverter::TTargetConverter(const EConvertTargetPolicy readingPoolTargetPolicy,
                                       const TVector<TString>& inputClassNames,
                                       TVector<TString>* const outputClassNames)
        : TargetPolicy(readingPoolTargetPolicy)
        , InputClassNames(inputClassNames)
        , OutputClassNames(outputClassNames)
    {
        if (TargetPolicy == EConvertTargetPolicy::MakeClassNames) {
            CB_ENSURE(outputClassNames != nullptr,
                      "Cannot initialize target converter with null class names pointer and MakeClassNames target policy.");
        }

        if (TargetPolicy == EConvertTargetPolicy::UseClassNames) {
            CB_ENSURE(!InputClassNames.empty(), "Cannot use empty class names for pool reading.");
            int id = 0;
            for (const auto& name : InputClassNames) {
                LabelToClass.emplace(name, id++);
            }
        }
    }


    float TTargetConverter::ConvertLabel(const TStringBuf& label) const {
        switch (TargetPolicy) {
            case EConvertTargetPolicy::CastFloat: {
                CB_ENSURE(!IsNanValue(label), "NaN not supported for target");
                return FromString<float>(label);
            }
            case EConvertTargetPolicy::UseClassNames: {
                const auto it = LabelToClass.find(label);
                if (it != LabelToClass.end()) {
                    return static_cast<float>(it->second);
                }
                ythrow TCatboostException() << "Unknown class name: " << label;
            }
            default: {
                ythrow TCatboostException() <<
                    "Cannot convert label online if convert target policy is not CastFloat or UseClassNames.";
            }
        }
    }


    float TTargetConverter::ProcessLabel(const TString& label) {
        THashMap<TString, int>::insert_ctx ctx = nullptr;
        const auto& it = LabelToClass.find(label, ctx);

        if (it == LabelToClass.end()) {
            const int classIdx = LabelToClass.ysize();
            LabelToClass.emplace_direct(ctx, label, classIdx);
            return static_cast<float>(classIdx);
        } else {
            return static_cast<float>(it->second);
        }
    }

    TVector<float> TTargetConverter::PostprocessLabels(TConstArrayRef<TString> labels) {
        CB_ENSURE(TargetPolicy == EConvertTargetPolicy::MakeClassNames,
                  "Cannot postprocess labels without MakeClassNames target policy.");
        THashSet<TString> uniqueLabelsSet(labels.begin(), labels.end());
        TVector<TString> uniqueLabels(uniqueLabelsSet.begin(), uniqueLabelsSet.end());
        Sort(uniqueLabels);
        CB_ENSURE(LabelToClass.empty(), "PostrpocessLabels: label-to-class map must be empty before label converting.");
        for (const auto& label: uniqueLabels) {
            ProcessLabel(label);
        }
        TVector<float> targets;
        targets.reserve(labels.size());
        for (const auto& label : labels) {
            targets.push_back(ProcessLabel(label));
        }
        return targets;
    }

    void TTargetConverter::SetOutputClassNames() const {
        CB_ENSURE(OutputClassNames != nullptr && OutputClassNames->empty(), "Cannot reset user-defined class names.");
        CB_ENSURE(TargetPolicy == EConvertTargetPolicy::MakeClassNames,
                  "Cannot set class names without MakeClassNames target policy.");
        CB_ENSURE(!LabelToClass.empty(), "Label-to-class mapping must be calced before setting class names.");
        OutputClassNames->resize(LabelToClass.ysize());
        for (const auto& keyValue : LabelToClass) {
            (*OutputClassNames)[keyValue.second] = keyValue.first;
        }
    }

    EConvertTargetPolicy TTargetConverter::GetTargetPolicy() const {
        return TargetPolicy;
    }

    const TVector<TString>& TTargetConverter::GetInputClassNames() const {
        return InputClassNames;
    }

    TTargetConverter MakeTargetConverter(const TVector<TString>& classNames) {
        return  TTargetConverter(classNames.empty() ?
                                   EConvertTargetPolicy::CastFloat :
                                   EConvertTargetPolicy::UseClassNames,
                                 classNames,
                                 nullptr
        );
    }

    THolder<IPoolBuilder> InitBuilder(
        const NCB::TPathWithScheme& poolPath,
        const NPar::TLocalExecutor& localExecutor,
        TPool* pool) {
        if (poolPath.Scheme == "quantized") {
            return new TQuantizedBuilder(pool);
        } else {
            return new TPoolBuilder(localExecutor, pool);
        }
    }

    void ReadPool(
        THolder<ILineDataReader> poolReader,
        const TPathWithScheme& pairsFilePath,
        const TPathWithScheme& groupWeightsFilePath,
        const NCB::TDsvFormatOptions& poolFormat,
        const TVector<TColumn>& columnsDescription, //TODO(smirnovpavel): EColumn is enough to build pool
        const TVector<int>& ignoredFeatures,
        const TVector<TString>& classNames,
        NPar::TLocalExecutor* localExecutor,
        TPool* pool
    ) {
        TPoolBuilder poolBuilder(*localExecutor, pool);
        TTargetConverter targetConverter = MakeTargetConverter(classNames);
        THolder<IDocPoolDataProvider> docPoolDataProvider = MakeHolder<TCBDsvDataProvider>(
            // processor args
            TDocPoolPushDataProviderArgs {
                std::move(poolReader),

                TDocPoolCommonDataProviderArgs {
                    pairsFilePath,
                    groupWeightsFilePath,
                    poolFormat,
                    MakeCdProviderFromArray(columnsDescription),
                    ignoredFeatures,
                    10000, // TODO: make it a named constant
                    &targetConverter,
                    localExecutor
                }
            }

        );
        docPoolDataProvider->Do(&poolBuilder);
    }

    void ReadPool(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<int>& ignoredFeatures,
        int threadCount,
        bool verbose,
        TPool* pool
    ) {
        TTargetConverter targetConverter(EConvertTargetPolicy::CastFloat, {}, nullptr);
        ReadPool(
            poolPath,
            pairsFilePath,
            groupWeightsFilePath,
            dsvPoolFormatParams,
            ignoredFeatures,
            threadCount,
            verbose,
            &targetConverter,
            pool
        );
    }

    void ReadPool(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const TPathWithScheme& groupWeightsFilePath,
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<int>& ignoredFeatures,
        int threadCount,
        bool verbose,
        TTargetConverter* const targetConverter,
        TPool* pool
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);
        THolder<IPoolBuilder> builder = InitBuilder(poolPath, localExecutor, pool);
        ReadPool(
            poolPath,
            pairsFilePath,
            groupWeightsFilePath,
            dsvPoolFormatParams,
            ignoredFeatures,
            verbose,
            targetConverter,
            &localExecutor,
            builder.Get()
        );
    }

    void ReadPool(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const TPathWithScheme& groupWeightsFilePath,
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<int>& ignoredFeatures,
        bool verbose,
        TTargetConverter* const targetConverter,
        NPar::TLocalExecutor* localExecutor,
        IPoolBuilder* poolBuilder
    ) {
        if (verbose) {
            SetVerboseLogingMode();
        } else {
            SetSilentLogingMode();
        }

        auto docPoolDataProvider = GetProcessor<IDocPoolDataProvider>(
            poolPath, // for choosing processor

            // processor args
            TDocPoolPullDataProviderArgs {
                poolPath,

                TDocPoolCommonDataProviderArgs {
                    pairsFilePath,
                    groupWeightsFilePath,
                    dsvPoolFormatParams.Format,
                    MakeCdProviderFromFile(dsvPoolFormatParams.CdFilePath),
                    ignoredFeatures,
                    10000, // TODO: make it a named constant
                    targetConverter,
                    localExecutor
                }
            }
        );

        docPoolDataProvider->Do(poolBuilder);

        SetVerboseLogingMode(); //TODO(smirnovpavel): verbose mode must be restored to initial
    }

    void ReadPool(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const TPathWithScheme& groupWeightsFilePath,
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        int threadCount,
        bool verbose,
        IPoolBuilder& poolBuilder
    ) {
        NPar::TLocalExecutor localExecutor;
        TTargetConverter targetConverter(EConvertTargetPolicy::CastFloat, {}, nullptr);
        localExecutor.RunAdditionalThreads(threadCount - 1);
        ReadPool(
            poolPath,
            pairsFilePath,
            groupWeightsFilePath,
            dsvPoolFormatParams,
            {},
            verbose,
            &targetConverter,
            &localExecutor,
            &poolBuilder
        );
    }
    void ReadTrainPools(
        const NCatboostOptions::TPoolLoadParams& loadOptions,
        bool readTestData,
        int threadCount,
        NCB::TTargetConverter* const trainTargetConverter,
        TMaybe<TProfileInfo*> profile,
        TTrainPools* trainPools
    ) {
        loadOptions.Validate();

        const bool verbose = false;
        if (loadOptions.LearnSetPath.Inited()) {
            ReadPool(
                loadOptions.LearnSetPath,
                loadOptions.PairsFilePath,
                loadOptions.GroupWeightsFilePath,
                loadOptions.DsvPoolFormatParams,
                loadOptions.IgnoredFeatures,
                threadCount,
                verbose,
                trainTargetConverter,
                &(trainPools->Learn)
            );
            if (profile) {
                (*profile)->AddOperation("Build learn pool");
            }
        }
        trainPools->Test.resize(0);

        if (readTestData) {
            const auto& trainClassNames = trainTargetConverter->GetInputClassNames();
            TTargetConverter testTargetConverter = MakeTargetConverter(trainClassNames);

            for (int testIdx = 0; testIdx < loadOptions.TestSetPaths.ysize(); ++testIdx) {
                const NCB::TPathWithScheme& testSetPath = loadOptions.TestSetPaths[testIdx];
                const NCB::TPathWithScheme& testPairsFilePath =
                        testIdx == 0 ? loadOptions.TestPairsFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testGroupWeightsFilePath =
                        testIdx == 0 ? loadOptions.TestGroupWeightsFilePath : NCB::TPathWithScheme();

                TPool testPool;
                ReadPool(
                    testSetPath,
                    testPairsFilePath,
                    testGroupWeightsFilePath,
                    loadOptions.DsvPoolFormatParams,
                    loadOptions.IgnoredFeatures,
                    threadCount,
                    verbose,
                    &testTargetConverter,
                    &testPool
                );
                trainPools->Test.push_back(std::move(testPool));
                if (profile.Defined() && (testIdx + 1 == loadOptions.TestSetPaths.ysize())) {
                    (*profile)->AddOperation("Build test pool");
                }
            }
        }
    }
} // NCB
