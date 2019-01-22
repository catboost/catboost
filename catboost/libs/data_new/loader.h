#pragma once

#include "async_row_processor.h"
#include "meta_info.h"
#include "objects.h"
#include "visitor.h"

#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/options/load_options.h>
#include <catboost/libs/column_description/cd_parser.h>

#include <library/object_factory/object_factory.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

namespace NCB {

    struct TDatasetLoaderCommonArgs {
        TPathWithScheme PairsFilePath;
        TPathWithScheme GroupWeightsFilePath;
        TDsvFormatOptions PoolFormat;
        THolder<ICdProvider> CdProvider;
        TVector<ui32> IgnoredFeatures;
        EObjectsOrder ObjectsOrder;
        ui32 BlockSize;
        NPar::TLocalExecutor* LocalExecutor;
    };

    // pass this struct to to IDatasetLoader ctor
    struct TDatasetLoaderPullArgs {
        TPathWithScheme PoolPath;
        TDatasetLoaderCommonArgs CommonArgs;
    };

    // pass this struct to to IDatasetLoader ctor
    struct TLineDataLoaderPushArgs {
        THolder<ILineDataReader> Reader;
        TDatasetLoaderCommonArgs CommonArgs;
    };


    struct IDatasetLoader {
        virtual ~IDatasetLoader() = default;

        virtual EDatasetVisitorType GetVisitorType() const = 0;

        /* Process all data
         *
         * checks dynamically that IDatasetVisitor is of compatible derived type
         * otherwise fails
         */
        virtual void DoIfCompatible(IDatasetVisitor* visitor) = 0;
    };

    struct IRawObjectsOrderDatasetLoader : public IDatasetLoader {
        virtual EDatasetVisitorType GetVisitorType() const override {
            return EDatasetVisitorType::RawObjectsOrder;
        }

        void DoIfCompatible(IDatasetVisitor* visitor) override {
            auto compatibleVisitor = dynamic_cast<IRawObjectsOrderDataVisitor*>(visitor);
            CB_ENSURE_INTERNAL(compatibleVisitor, "visitor is incompatible with dataset loader");
            Do(compatibleVisitor);
        }

        // Process all data
        virtual void Do(IRawObjectsOrderDataVisitor* visitor) = 0;

        // Process next block of objects and build sub-pool from it
        // returns true if any rows were processed
        virtual bool DoBlock(IRawObjectsOrderDataVisitor* visitor) = 0;
    };

    struct IRawFeaturesOrderDatasetLoader : public IDatasetLoader {
        virtual EDatasetVisitorType GetVisitorType() const override {
            return EDatasetVisitorType::RawObjectsOrder;
        }

        // Process all data
        virtual void Do(IRawFeaturesOrderDataVisitor* visitor) = 0;
    };

    struct IQuantizedFeaturesDatasetLoader : public IDatasetLoader {
        virtual EDatasetVisitorType GetVisitorType() const override {
            return EDatasetVisitorType::QuantizedFeatures;
        }

        void DoIfCompatible(IDatasetVisitor* visitor) override {
            auto compatibleVisitor = dynamic_cast<IQuantizedFeaturesDataVisitor*>(visitor);
            CB_ENSURE_INTERNAL(compatibleVisitor, "visitor is incompatible with dataset loader");
            Do(compatibleVisitor);
        }

        // Process all data
        virtual void Do(IQuantizedFeaturesDataVisitor* visitor) = 0;

        // TODO(akhropov): support blocks by features, by docs?
    };


    using TDatasetLoaderFactory =
        NObjectFactory::TParametrizedObjectFactory<IDatasetLoader,
                                                   TString,
                                                   TDatasetLoaderPullArgs>;



    ///////////////////////////////////////////////////////////////////////////
    // Common functionality used in IDatasetLoader implementations

    /*
     * init ignored features information in dataMetaInfo and ignoredFeaturesMask
     * from ignoredFeaturesFlatIndices
     *
     * note that ignoredFeaturesFlatIndices can contain indices beyond featuresCount in dataMetaInfo
     */
    void ProcessIgnoredFeaturesList(
        TConstArrayRef<ui32> ignoredFeatures, // [flatFeatureIdx]
        TDataMetaInfo* dataMetaInfo, // inout, must be inited, only ignored flags are updated
        TVector<bool>* ignoredFeaturesMask // [flatFeatureIdx]
    );


    void SetPairs(const TPathWithScheme& pairsPath, ui32 objectCount, IDatasetVisitor* visitor);
    void SetGroupWeights(
        const TPathWithScheme& groupWeightsPath,
        ui32 objectCount,
        IDatasetVisitor* visitor
    );

    /*
     * Some common functionality for IRawObjectsOrderDatasetLoader classes than utilize async row processing.
     *   Args, FeatureIds and DataMetaInfo are provided as commonly needed
     *   (but not related to async processing)
     *
     *  Derived classes must implement GetObjectDount, StartBuilder and ProcessBlock
     *  (and might redefine FinalizeBuilder, but common implementation is provided)
     *  and then implement IRawObjectsOrderDatasetLoader like this:
     *
     * > void Do(IRawObjectsOrderDataVisitor* visitor) override {
     * >      TBase::Do(GetReadFunc(), visitor);
     * >  }
     * >
     * >  bool DoBlock(IRawObjectsOrderDataVisitor* visitor) override {
     * >      return TBase::DoBlock(GetReadFunc(), visitor);
     * >  }
     */
    template <class TData>
    class TAsyncProcDataLoaderBase {
    public:
        explicit TAsyncProcDataLoaderBase(TDatasetLoaderCommonArgs&& args)
            : Args(std::move(args))
            , AsyncRowProcessor(Args.LocalExecutor, Args.BlockSize)
        {}

    protected:
        template <class TReadDataFunc>
        void Do(TReadDataFunc readFunc, IRawObjectsOrderDataVisitor* visitor) {
            StartBuilder(false, GetObjectCount(), 0, visitor);
            while (AsyncRowProcessor.ReadBlock(readFunc)) {
                ProcessBlock(visitor);
            }
            FinalizeBuilder(false, visitor);
        }

        template <class TReadDataFunc>
        bool DoBlock(TReadDataFunc readFunc, IRawObjectsOrderDataVisitor* visitor) {
            CB_ENSURE(!Args.PairsFilePath.Inited(),
                      "TAsyncProcDataLoaderBase::DoBlock does not support pairs data");
            CB_ENSURE(!Args.GroupWeightsFilePath.Inited(),
                      "TAsyncProcDataLoaderBase::DoBlock does not support group weights data");

            if (!AsyncRowProcessor.ReadBlock(readFunc))
                return false;

            StartBuilder(true,
                         AsyncRowProcessor.GetParseBufferSize(),
                         AsyncRowProcessor.GetLinesProcessed(),
                         visitor);
            ProcessBlock(visitor);
            FinalizeBuilder(true, visitor);

            return true;
        }


        virtual ui32 GetObjectCount() = 0;

        virtual void StartBuilder(bool inBlock,
                                  ui32 objectCount, ui32 offset,
                                  IRawObjectsOrderDataVisitor* visitor) = 0;

        virtual void ProcessBlock(IRawObjectsOrderDataVisitor* visitor) = 0;

        virtual void FinalizeBuilder(bool inBlock, IRawObjectsOrderDataVisitor* visitor) {
            if (!inBlock) {
                SetGroupWeights(Args.GroupWeightsFilePath, GetObjectCount(), visitor);
                SetPairs(Args.PairsFilePath, GetObjectCount(), visitor);
            }
            visitor->Finish();
        }

        virtual ~TAsyncProcDataLoaderBase() = default;

    protected:
        TDatasetLoaderCommonArgs Args;
        NCB::TAsyncRowProcessor<TData> AsyncRowProcessor;

        TVector<TString> FeatureIds;
        TDataMetaInfo DataMetaInfo;
    };

    /* missing values mostly as in pandas
     * (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)
     *  + some additions like 'NAN', 'Na', 'na', 'Null', 'none', 'None', '-'
     */
    bool IsMissingValue(const TStringBuf& s);

    bool TryParseFloatFeatureValue(TStringBuf stringValue, float* value);
}
