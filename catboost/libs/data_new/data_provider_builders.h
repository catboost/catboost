#pragma once

#include "data_provider.h"
#include "loader.h"
#include "quantized_features_info.h"
#include "visitor.h"

#include <catboost/libs/helpers/exception.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>

#include <functional>


namespace NCB {

    // hack to extract private data from inside providers
    class TRawBuilderDataHelper {
    public:
        static TRawBuilderData Extract(TRawDataProvider&& rawDataProvider) {
            TRawBuilderData data;
            data.MetaInfo = std::move(rawDataProvider.MetaInfo);
            data.TargetData = std::move(rawDataProvider.RawTargetData.Data);
            data.CommonObjectsData = std::move(rawDataProvider.ObjectsData->CommonData);
            data.ObjectsData = std::move(rawDataProvider.ObjectsData->Data);
            return data;
        }
    };

    struct IDataProviderBuilder {
        virtual ~IDataProviderBuilder() = default;

        virtual TDataProviderPtr GetResult() = 0;

        /* can return nullptr, needed to get last group data when processing is by blocks,
         * call after last GetResult
         */
        virtual TDataProviderPtr GetLastResult() { return nullptr; }
    };

    struct TDataProviderBuilderOptions {
        bool CpuCompatibleFormat = true;
        bool GpuCompatibleFormat = true;
        bool SkipCheck = false; // to increase speed, esp. when applying
    };

    // can return nullptr if IDataProviderBuilder for such visitor type hasn't been implemented yet
    THolder<IDataProviderBuilder> CreateDataProviderBuilder(
        EDatasetVisitorType visitorType,
        const TDataProviderBuilderOptions& options,
        TDatasetSubset loadSubset,
        NPar::TLocalExecutor* localExecutor
    );


    class TDataProviderClosure : public IDataProviderBuilder
    {
    public:
        TDataProviderClosure(
            EDatasetVisitorType visitorType,
            const TDataProviderBuilderOptions& options
        ) {
            DataProviderBuilder = CreateDataProviderBuilder(
                visitorType,
                options,
                TDatasetSubset::MakeColumns(),
                &NPar::LocalExecutor()
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
        THolder<IDataProviderBuilder>* dataProviderBuilder,
        IVisitor** builderVisitor
    ) {
        auto dataProviderClosure = MakeHolder<TDataProviderClosure>(
            IVisitor::Type,
            options
        );
        *builderVisitor = dataProviderClosure->template GetVisitor<IVisitor>();
        *dataProviderBuilder = dataProviderClosure.Release();
    }

    /*
     * uses global LocalExecutor inside
     * call IVisitor's methods in loader
     */
    template <class IVisitor = IRawFeaturesOrderDataVisitor, class TLoader>
    TDataProviderPtr CreateDataProvider(
        TLoader&& loader,
        const TDataProviderBuilderOptions& options = TDataProviderBuilderOptions()
    ) {
        TDataProviderClosure dataProviderClosure(IVisitor::Type, options);
        loader(dataProviderClosure.GetVisitor<IVisitor>());
        return dataProviderClosure.GetResult();
    }

    /* common case
     * floatFeatures is [objectIdx][featureIdx] matrix, skip data check for speed
     */
    TDataProviderPtr CreateDataProviderFromObjectsOrderData(
        TVector<TVector<float>>&& floatFeatures
    );
}
