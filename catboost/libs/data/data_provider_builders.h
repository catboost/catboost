#pragma once

#include "data_provider.h"
#include "loader.h"
#include "quantized_features_info.h"
#include "visitor.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/sparse_array.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>

#include <functional>


namespace NCB {

    // hack to extract private data from inside providers
    template <class TTObjectsDataProvider>
    class TBuilderDataHelper {
        using TTDataProvider = TDataProviderTemplate<TTObjectsDataProvider>;
        using TTBuilderData = TBuilderData<decltype(std::declval<TTDataProvider>().ObjectsData->ExtractObjectData())>;
    public:
        static TTBuilderData Extract(TTDataProvider&& dataProvider) {
            TTBuilderData data;
            data.MetaInfo = std::move(dataProvider.MetaInfo);
            data.TargetData = std::move(dataProvider.RawTargetData.Data);
            data.CommonObjectsData = std::move(dataProvider.ObjectsData->CommonData);
            data.ObjectsData = dataProvider.ObjectsData->ExtractObjectData();
            return data;
        }
    };

    using TRawBuilderDataHelper = TBuilderDataHelper<TRawObjectsDataProvider>;
    using TQuantizedBuilderDataHelper = TBuilderDataHelper<TQuantizedObjectsDataProvider>;

    struct IDataProviderBuilder {
        virtual ~IDataProviderBuilder() = default;

        /* can return nullptr when processing is by blocks, it means after processing current block no new
         * complete groups found, adding data from subsequent blocks is required
         */
        virtual TDataProviderPtr GetResult() = 0;

        /* can return nullptr, needed to get last group data when processing is by blocks,
         * call after last GetResult
         */
        virtual TDataProviderPtr GetLastResult() { return nullptr; }
    };

    struct TDataProviderBuilderOptions {
        bool GpuDistributedFormat = false;
        TPathWithScheme PoolPath = TPathWithScheme();
        ui64 MaxCpuRamUsage = Max<ui64>();
        bool SkipCheck = false; // to increase speed, esp. when applying
        ESparseArrayIndexingType SparseArrayIndexingType = ESparseArrayIndexingType::Undefined;
    };

    // can return nullptr if IDataProviderBuilder for such visitor type hasn't been implemented yet
    THolder<IDataProviderBuilder> CreateDataProviderBuilder(
        EDatasetVisitorType visitorType,
        const TDataProviderBuilderOptions& options,
        TDatasetSubset loadSubset,
        NPar::ILocalExecutor* localExecutor
    );


    class TDataProviderClosure : public IDataProviderBuilder
    {
    public:
        TDataProviderClosure(
            EDatasetVisitorType visitorType,
            const TDataProviderBuilderOptions& options,
            NPar::ILocalExecutor* localExecutor
        ) {
            DataProviderBuilder = CreateDataProviderBuilder(
                visitorType,
                options,
                TDatasetSubset::MakeColumns(),
                localExecutor
            );
            CB_ENSURE_INTERNAL(
                DataProviderBuilder.Get(),
                "Failed to create data provider builder for visitor of type "
                << visitorType
            );
        }

        template <class IVisitor>
        IVisitor* GetVisitor() {
            return dynamic_cast<IVisitor*>(DataProviderBuilder.Get());
        }

        TDataProviderPtr GetResult() override {
            return DataProviderBuilder->GetResult();
        }

    private:
        THolder<IDataProviderBuilder> DataProviderBuilder;
    };


    /*
     * call builderVisitor's methods in loader
     * then call GetResult of dataProvider
     *
     * needed for Cython
     */
    template <class IVisitor>
    void CreateDataProviderBuilderAndVisitor(
        const TDataProviderBuilderOptions& options,
        NPar::ILocalExecutor* localExecutor,
        THolder<IDataProviderBuilder>* dataProviderBuilder,
        IVisitor** builderVisitor
    ) {
        auto dataProviderClosure = MakeHolder<TDataProviderClosure>(
            IVisitor::Type,
            options,
            localExecutor
        );
        *builderVisitor = dataProviderClosure->template GetVisitor<IVisitor>();
        *dataProviderBuilder = std::move(dataProviderClosure);
    }

    /*
     * uses global LocalExecutor inside
     * call IVisitor's methods in loader
     *
     * needed by unit tests and R-package
     */
    template <class IVisitor = IRawFeaturesOrderDataVisitor, class TLoader>
    TDataProviderPtr CreateDataProvider(
        TLoader&& loader,
        const TDataProviderBuilderOptions& options = TDataProviderBuilderOptions()
    ) {
        TDataProviderClosure dataProviderClosure(IVisitor::Type, options, &NPar::LocalExecutor());
        loader(dataProviderClosure.GetVisitor<IVisitor>());
        return dataProviderClosure.GetResult();
    }
}
