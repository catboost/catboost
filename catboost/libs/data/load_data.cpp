#include "load_data.h"

#include "doc_pool_data_provider.h"

#include <catboost/libs/helpers/exception.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/hash.h>


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

        int GetDocCount() const override {
            return NextCursor;
        }

        TConstArrayRef<float> GetWeight() const override {
            return MakeArrayRef(Pool->Docs.Weight.data(), Pool->Docs.Weight.size());
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

    }

    THolder<IPoolBuilder> InitBuilder(const NPar::TLocalExecutor& localExecutor, TPool* pool) {
        return new TPoolBuilder(localExecutor, pool);
    }

    void ReadPool(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<int>& ignoredFeatures,
        int threadCount,
        bool verbose,
        const TVector<TString>& classNames,
        TPool* pool
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);
        TPoolBuilder builder(localExecutor, pool);
        ReadPool(
            poolPath,
            pairsFilePath,
            dsvPoolFormatParams,
            ignoredFeatures,
            verbose,
            classNames,
            &localExecutor,
            &builder
        );
    }

    void ReadPool(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<int>& ignoredFeatures,
        bool verbose,
        const TVector<TString>& classNames,
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
            TDocPoolDataProviderArgs {
                 poolPath,
                 pairsFilePath,
                 dsvPoolFormatParams,
                 ignoredFeatures,
                 classNames,
                 10000, // TODO: make it a named constant
                 localExecutor
            }
        );

        docPoolDataProvider->Do(poolBuilder);

        SetVerboseLogingMode();
    }

    void ReadPool(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        int threadCount,
        bool verbose,
        IPoolBuilder& poolBuilder
    ) {
        TVector<TString> noNames;
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);
        ReadPool(poolPath, pairsFilePath, dsvPoolFormatParams, {}, verbose, noNames, &localExecutor, &poolBuilder);
    }

}
