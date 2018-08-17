#pragma once

#include "async_row_processor.h"

#include "load_data.h"

#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/options/load_options.h>
#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/pool_builder/pool_builder.h>

#include <library/object_factory/object_factory.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>

namespace NCB {

    struct TDocPoolCommonDataProviderArgs {
        TPathWithScheme PairsFilePath;
        TPathWithScheme GroupWeightsFilePath;
        TDsvFormatOptions PoolFormat;
        THolder<ICdProvider> CdProvider;
        TVector<int> IgnoredFeatures;
        ui32 BlockSize;
        TTargetConverter* TargetConverter;
        NPar::TLocalExecutor* LocalExecutor;
    };

    // pass this struct to to IDocPoolDataProvider ctor
    struct TDocPoolPullDataProviderArgs {
        TPathWithScheme PoolPath;
        TDocPoolCommonDataProviderArgs CommonArgs;
    };

    // pass this struct to to IDocPoolDataProvider ctor
    struct TDocPoolPushDataProviderArgs {
        THolder<ILineDataReader> PoolReader;
        TDocPoolCommonDataProviderArgs CommonArgs;
    };


    struct IDocPoolDataProvider {
        // Process all data
        virtual void Do(IPoolBuilder* poolBuilder) = 0;

        // Process next block of docs and build sub-pool from it
        // returns true if any rows were processed
        virtual bool DoBlock(IPoolBuilder* poolBuilder) = 0;

        virtual ~IDocPoolDataProvider() = default;
    };

    using TDocDataProviderObjectFactory =
        NObjectFactory::TParametrizedObjectFactory<IDocPoolDataProvider,
                                                   TString,
                                                   TDocPoolPullDataProviderArgs>;



    ///////////////////////////////////////////////////////////////////////////
    // Implementations

    void WeightPairs(TConstArrayRef<float> groupWeight, TVector<TPair>* pairs);
    void SetPairs(const TPathWithScheme& pairsPath, bool haveGroupWeights, IPoolBuilder* poolBuilder);
    void SetGroupWeights(const TPathWithScheme& groupWeightsPath, IPoolBuilder* poolBuilder);

    /*
     * Some common functionality for DocPoolDataProvider classes than utilize async row processing.
     *   Args, FeatureIds and PoolMetaInfo are provided as commonly needed
     *   (but not related to async processing)
     *
     *  Derived classes must implement GetDocDount, StartBuilder and ProcessBlock
     *  (and might redefine FinalizeBuilder, but common implementation is provided)
     *  and then implement IDocPoolDataProvider like this:
     *
     * > void Do(IPoolBuilder* poolBuilder) override {
     * >      TBase::Do(GetReadFunc(), poolBuilder);
     * >  }
     * >
     * >  bool DoBlock(IPoolBuilder* poolBuilder) override {
     * >      return TBase::DoBlock(GetReadFunc(), poolBuilder);
     * >  }
     */
    template <class TData>
    class TAsyncProcDataProviderBase {
    public:
        explicit TAsyncProcDataProviderBase(TDocPoolCommonDataProviderArgs&& args)
            : Args(std::move(args))
            , AsyncRowProcessor(Args.LocalExecutor, Args.BlockSize)
            , IsOfflineTargetProcessing(false)
            , IsOnlineTargetProcessing(false)
            , TargetConverter(Args.TargetConverter)
        {
            CB_ENSURE(Args.TargetConverter != nullptr,
                "TAsyncProcDataProviderBase can not work with null target converter pointer");
        }

    protected:
        template <class TReadDataFunc>
        void Do(TReadDataFunc readFunc, IPoolBuilder* poolBuilder) {
            StartBuilder(false, GetDocCount(), 0, poolBuilder);
            while (AsyncRowProcessor.ReadBlock(readFunc)) {
                ProcessBlock(poolBuilder);
            }
            FinalizeBuilder(false, poolBuilder);
        }

        template <class TReadDataFunc>
        bool DoBlock(TReadDataFunc readFunc, IPoolBuilder* poolBuilder) {
            CB_ENSURE(!Args.PairsFilePath.Inited(),
                      "TAsyncProcDataProviderBase::DoBlock does not support pairs data");
            CB_ENSURE(!Args.GroupWeightsFilePath.Inited(),
                      "TAsyncProcDataProviderBase::DoBlock does not support group weights data");

            if (!AsyncRowProcessor.ReadBlock(readFunc))
                return false;

            StartBuilder(true,
                         AsyncRowProcessor.GetParseBufferSize(),
                         AsyncRowProcessor.GetLinesProcessed(),
                         poolBuilder);
            ProcessBlock(poolBuilder);
            FinalizeBuilder(true, poolBuilder);

            return true;
        }


        virtual int GetDocCount() = 0;

        virtual void StartBuilder(bool inBlock,
                                  int docCount, int offset,
                                  IPoolBuilder* poolBuilder) = 0;

        virtual void ProcessBlock(IPoolBuilder* poolBuilder) = 0;

        virtual void FinalizeBuilder(bool inBlock, IPoolBuilder* poolBuilder) {
            if (!inBlock) {
                SetGroupWeights(Args.GroupWeightsFilePath, poolBuilder);
                SetPairs(Args.PairsFilePath, PoolMetaInfo.HasGroupWeight, poolBuilder);
                if (IsOfflineTargetProcessing) {
                    poolBuilder->SetTarget(TargetConverter->PostprocessLabels(poolBuilder->GetLabels()));  // postprocessing is used in order to save class-names reproducibility for multithreading
                    TargetConverter->SetOutputClassNames();
                }
            }
            poolBuilder->Finish();
        }

        virtual ~TAsyncProcDataProviderBase() = default;

    protected:
        TDocPoolCommonDataProviderArgs Args;
        NCB::TAsyncRowProcessor<TData> AsyncRowProcessor;

        bool IsOfflineTargetProcessing;
        bool IsOnlineTargetProcessing;

        TTargetConverter* TargetConverter;
        TVector<TString> FeatureIds;
        TPoolMetaInfo PoolMetaInfo;
    };


    // expose the declaration to allow to derive from it in other modules
    class TCBDsvDataProvider : public IDocPoolDataProvider
                             , protected TAsyncProcDataProviderBase<TString>
    {
    public:
        using TBase = TAsyncProcDataProviderBase<TString>;

    protected:
        decltype(auto) GetReadFunc() {
            return [this](TString* line) -> bool {
                return LineDataReader->ReadLine(line);
            };
        }

    public:
        explicit TCBDsvDataProvider(TDocPoolPullDataProviderArgs&& args);

        explicit TCBDsvDataProvider(TDocPoolPushDataProviderArgs&& args);

        ~TCBDsvDataProvider() {
            AsyncRowProcessor.FinishAsyncProcessing();
        }

        void Do(IPoolBuilder* poolBuilder) override {
            TBase::Do(GetReadFunc(), poolBuilder);
        }

        bool DoBlock(IPoolBuilder* poolBuilder) override {
            return TBase::DoBlock(GetReadFunc(), poolBuilder);
        }

        TVector<TColumn> CreateColumnsDescription(ui32 columnsCount);

        int GetDocCount() override {
            return (int)LineDataReader->GetDataLineCount();
        }

        void StartBuilder(bool inBlock, int docCount, int offset, IPoolBuilder* poolBuilder) override;

        void ProcessBlock(IPoolBuilder* poolBuilder) override;

    protected:
        TVector<bool> FeatureIgnored; // init in process
        char FieldDelimiter;
        THolder<NCB::ILineDataReader> LineDataReader;

        TVector<int> CatFeatures;
    };


    bool IsNanValue(const TStringBuf& s);
}
