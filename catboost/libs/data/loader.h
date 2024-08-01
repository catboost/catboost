#pragma once

#include "async_row_processor.h"
#include "baseline.h"
#include "meta_info.h"
#include "objects.h"
#include "visitor.h"

#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/data_types/pair.h>
#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/private/libs/options/load_options.h>
#include <catboost/libs/column_description/cd_parser.h>

#include <library/cpp/object_factory/object_factory.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>


namespace NJson {
    class TJsonValue;
}


namespace NCB {

    struct TDatasetSubset {
        bool HasFeatures = true;
        TIndexRange<ui64> Range = {0, Max<ui64>()};

    public:
        ui64 GetSize() const { return Range.GetSize(); }

        static TDatasetSubset MakeRange(ui64 start, ui64 end) {
            return {true, {start, end}};
        }

        static TDatasetSubset MakeColumns(bool hasFeatures = true) {
            return {hasFeatures, {0u, Max<ui64>()}};
        }

        size_t GetHash() const {
            return MultiHash(HasFeatures, Range.Begin, Range.End);
        }

        bool operator==(const TDatasetSubset& rhs) const {
            return std::tie(HasFeatures, Range) == std::tie(rhs.HasFeatures, rhs.Range);
        }

        bool operator!=(const TDatasetSubset& rhs) const {
            return !(rhs == *this);
        }
    };

    struct TDatasetLoaderCommonArgs {
        TPathWithScheme PairsFilePath;
        TPathWithScheme GraphFilePath;
        TPathWithScheme GroupWeightsFilePath;
        TPathWithScheme BaselineFilePath;
        TPathWithScheme TimestampsFilePath;
        TPathWithScheme FeatureNamesPath;
        TPathWithScheme PoolMetaInfoPath;
        const TVector<NJson::TJsonValue>& ClassLabels;
        TDsvFormatOptions PoolFormat;
        THolder<ICdProvider> CdProvider;
        TVector<ui32> IgnoredFeatures;
        EObjectsOrder ObjectsOrder = EObjectsOrder::Undefined;
        ui32 BlockSize = 0;
        TDatasetSubset DatasetSubset;
        bool LoadColumnsAsString = false;
        bool LoadSampleIds = false; // special flag because they are rarely used
        bool ForceUnitAutoPairWeights = false;
        NPar::ILocalExecutor* LocalExecutor = nullptr;
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

    using TDatasetLineDataLoaderFactory =
        NObjectFactory::TParametrizedObjectFactory<IDatasetLoader,
                                                   TString,
                                                   TLineDataLoaderPushArgs>;

    ///////////////////////////////////////////////////////////////////////////
    // Common functionality used in IDatasetLoader implementations

    /*
     * init ignored features information in dataMetaInfo and ignoredFeaturesMask
     * from ignoredFeaturesFlatIndices
     *
     * it is possible to redefine default message when all features are ignored by specifying non-Nothing()
     *   allFeaturesIgnoredMessage parameter
     *
     * note that ignoredFeaturesFlatIndices can contain indices beyond featuresCount in dataMetaInfo
     */
    void ProcessIgnoredFeaturesList(
        TConstArrayRef<ui32> ignoredFeatures, // [flatFeatureIdx]
        TMaybe<TString> allFeaturesIgnoredMessage,
        TDataMetaInfo* dataMetaInfo, // inout, must be inited, only ignored flags are updated
        TVector<bool>* ignoredFeaturesMask // [flatFeatureIdx]
    );


    /*
     * Set pairs/group weights/baseline from loadSubset.Range of data
     * Indices of objects passed to visitor methods are indices from the beginning of the subset (not indices in the whole dataset).
     * objectCount parameter represents the number of objects in the subset.
     */
    THashMap<TGroupId, ui32> ConvertGroupIdToIdxMap(TConstArrayRef<TGroupId> groupIdsArray);
    void SetPairs(
        const TPathWithScheme& pairsPath,
        TDatasetSubset loadSubset,
        IDatasetVisitor* visitor);
    void SetGraph(
        const TPathWithScheme& graphPath,
        TDatasetSubset loadSubset,
        IDatasetVisitor* visitor);
    void SetGroupWeights(
        const TPathWithScheme& groupWeightsPath,
        ui32 objectCount,
        TDatasetSubset loadSubset,
        IDatasetVisitor* visitor
    );
    void SetBaseline(
        const TPathWithScheme& baselinePath,
        ui32 objectCount,
        TDatasetSubset loadSubset,
        const TVector<TString>& classNames,
        IDatasetVisitor* visitor
    );
    void SetTimestamps(
        const TPathWithScheme& timestampsPath,
        ui32 objectCount,
        TDatasetSubset loadSubset,
        IDatasetVisitor* visitor
    );

    // returns empty vector if featureNamesPath is not inited
    TVector<TString> LoadFeatureNames(const TPathWithScheme& featureNamesPath);

    TVector<TString> GetFeatureNames(
        const TDataColumnsMetaInfo& columnsDescription,
        const TMaybe<TVector<TString>>& headerColumns,
        const TPathWithScheme& featureNamesPath // can be uninited
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
            , AsyncBaselineRowProcessor(Args.LocalExecutor, Args.BlockSize)
        {}

    protected:
        template <class TReadDataFunc, class TReadBaselineFunc>
        void Do(TReadDataFunc readFunc, TReadBaselineFunc readBaselineFunc, IRawObjectsOrderDataVisitor* visitor) {
            StartBuilder(false, GetObjectCountSynchronized(), 0, visitor);
            while (AsyncRowProcessor.ReadBlock(readFunc)) {
                CB_ENSURE(!Args.BaselineFilePath.Inited() || AsyncBaselineRowProcessor.ReadBlock(readBaselineFunc), "Failed to read baseline");
                ProcessBlock(visitor);
            }
            FinalizeBuilder(false, visitor);
        }

        template <class TReadDataFunc, class TReadBaselineFunc>
        bool DoBlock(TReadDataFunc readFunc, TReadBaselineFunc readBaselineFunc, IRawObjectsOrderDataVisitor* visitor) {
            CB_ENSURE(!Args.PairsFilePath.Inited(),
                      "TAsyncProcDataLoaderBase::DoBlock does not support pairs data");
            CB_ENSURE(!Args.GroupWeightsFilePath.Inited(),
                      "TAsyncProcDataLoaderBase::DoBlock does not support group weights data");

            if (!AsyncRowProcessor.ReadBlock(readFunc))
                return false;

            CB_ENSURE(!Args.BaselineFilePath.Inited() || AsyncBaselineRowProcessor.ReadBlock(readBaselineFunc), "Failed to read baseline");

            StartBuilder(true,
                         AsyncRowProcessor.GetParseBufferSize(),
                         AsyncRowProcessor.GetLinesProcessed(),
                         visitor);
            ProcessBlock(visitor);
            FinalizeBuilder(true, visitor);

            return true;
        }

        // Some implementations use caching with synchronized access
        virtual ui32 GetObjectCountSynchronized() = 0;

        virtual void StartBuilder(bool inBlock,
                                  ui32 objectCount, ui32 offset,
                                  IRawObjectsOrderDataVisitor* visitor) = 0;

        virtual void ProcessBlock(IRawObjectsOrderDataVisitor* visitor) = 0;

        virtual void FinalizeBuilder(bool inBlock, IRawObjectsOrderDataVisitor* visitor) {
            if (!inBlock) {
                const ui32 objectCount = GetObjectCountSynchronized();
                SetGroupWeights(Args.GroupWeightsFilePath, objectCount, Args.DatasetSubset, visitor);
                SetPairs(Args.PairsFilePath, Args.DatasetSubset, visitor);
                SetGraph(Args.GraphFilePath, Args.DatasetSubset, visitor);
                SetTimestamps(Args.TimestampsFilePath, objectCount, Args.DatasetSubset, visitor);
            }
            visitor->Finish();
        }

        virtual ~TAsyncProcDataLoaderBase() = default;

    protected:
        TDatasetLoaderCommonArgs Args;
        NCB::TAsyncRowProcessor<TData> AsyncRowProcessor;
        NCB::TAsyncRowProcessor<TObjectBaselineData> AsyncBaselineRowProcessor;

        TVector<TString> FeatureIds;
        TDataMetaInfo DataMetaInfo;
    };

    /* missing values mostly as in pandas
     * (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)
     *  + some additions like 'NAN', 'Na', 'na', 'Null', 'none', 'None', '-'
     */
    bool IsMissingValue(const TStringBuf& s);

    bool TryFloatFromString(TStringBuf token, bool parseNonFinite, float* value);
}

template <>
struct THash<NCB::TDatasetSubset> {
    inline size_t operator()(const NCB::TDatasetSubset& value) const {
        return value.GetHash();
    }
};
