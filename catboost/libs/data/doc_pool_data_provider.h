#pragma once

#include "async_row_processor.h"
#include "load_data.h"

#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/options/load_options.h>

#include <library/object_factory/object_factory.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>


namespace NCB {

    // pass this struct to to IDocPoolDataProvider ctor
    struct TDocPoolDataProviderArgs {
        TPathWithScheme PoolPath;
        TPathWithScheme PairsFilePath;
        NCatboostOptions::TDsvPoolFormatParams DsvPoolFormatParams;
        TVector<int> IgnoredFeatures;
        TVector<TString> ClassNames;
        ui32 BlockSize;
        NPar::TLocalExecutor* LocalExecutor;
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
                                                   TDocPoolDataProviderArgs>;



    ///////////////////////////////////////////////////////////////////////////
    // Implementations

    TVector<TPair> ReadPairs(const TPathWithScheme& filePath, int docCount);
    void WeightPairs(TConstArrayRef<float> groupWeight, TVector<TPair>* pairs);

    class TTargetConverter {
    public:
        static constexpr float UNDEFINED_CLASS = -1;

        explicit TTargetConverter(const TVector<TString>& classNames);

        float operator()(const TString& word) const;

    private:
        TVector<TString> ClassNames;
    };

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
        explicit TAsyncProcDataProviderBase(TDocPoolDataProviderArgs&& args)
            : Args(std::move(args))
            , AsyncRowProcessor(Args.LocalExecutor, Args.BlockSize)
        {}

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
                DumpMemUsage("After data read");
                if (Args.PairsFilePath.Inited()) {
                    TVector<TPair> pairs = ReadPairs(Args.PairsFilePath, poolBuilder->GetDocCount());
                    if (PoolMetaInfo.HasGroupWeight) {
                        WeightPairs(poolBuilder->GetWeight(), &pairs);
                    }
                    poolBuilder->SetPairs(pairs);
                }
            }
            poolBuilder->Finish();
        }

        virtual ~TAsyncProcDataProviderBase() = default;

    protected:
        TDocPoolDataProviderArgs Args;
        NCB::TAsyncRowProcessor<TData> AsyncRowProcessor;

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
        explicit TCBDsvDataProvider(TDocPoolDataProviderArgs&& args);

        void Do(IPoolBuilder* poolBuilder) override {
            TBase::Do(GetReadFunc(), poolBuilder);
        }

        bool DoBlock(IPoolBuilder* poolBuilder) override {
            return TBase::DoBlock(GetReadFunc(), poolBuilder);
        }

        TVector<TColumn> CreateColumnsDescription(ui32 columnsCount);

        // call after ColumnDescription initialization
        void InitFeatureIds(const TMaybe<TString>& header);

        int GetDocCount() override {
            return (int)LineDataReader->GetDataLineCount();
        }

        void StartBuilder(bool inBlock, int docCount, int offset, IPoolBuilder* poolBuilder) override;

        void ProcessBlock(IPoolBuilder* poolBuilder) override;

    protected:
        TVector<bool> FeatureIgnored; // init in process
        char FieldDelimiter;
        TTargetConverter ConvertTarget;
        THolder<NCB::ILineDataReader> LineDataReader;

        TVector<int> CatFeatures;
    };

}
