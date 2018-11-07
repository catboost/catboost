#pragma once

#include "data_provider.h"
#include "quantized_features_info.h"
#include "visitor.h"

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>


namespace NCB {

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
    };

    // can return nullptr if IDataProviderBuilder for such visitor type hasn't been implemented yet
    THolder<IDataProviderBuilder> CreateDataProviderBuilder(
        EDatasetVisitorType visitorType,
        const TDataProviderBuilderOptions& options,
        NPar::TLocalExecutor* localExecutor
    );

}
