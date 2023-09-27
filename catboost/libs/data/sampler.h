#pragma once

#include "data_provider.h"

#include <catboost/private/libs/options/dataset_reading_params.h>

#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/ylimits.h>
#include <util/system/types.h>


template <class T, class D>
class THolder;

struct TColumn;

namespace MPar {
    class ILocalExecutor;
}


namespace NCB {
    struct TPathWithScheme;

    struct TDataProviderSampleParams {
        NCatboostOptions::TDatasetReadingParams DatasetReadingParams;
        bool OnlyFeaturesData = false;
        ui64 CpuUsedRamLimit = Max<ui64>();
        NPar::ILocalExecutor* LocalExecutor = nullptr; // always pass it, default value here is only for consistency
    };

    struct IDataProviderSampler {
        virtual ~IDataProviderSampler() {}

        virtual TDataProviderPtr SampleByIndices(TConstArrayRef<ui32> indices) = 0;
        virtual TDataProviderPtr SampleBySampleIds(TConstArrayRef<TString> sampleIds) = 0;
    };

    using TDataProviderSamplerFactory
        = NObjectFactory::TParametrizedObjectFactory<IDataProviderSampler, TString, TDataProviderSampleParams>;


    /* note that some formats don't use column description,
       so even if they contain sampleId data sampleIdColumnIdx may be ununitialized
    */
    void ReadCDForSampler(
        const TPathWithScheme& cdPath,
        bool onlyFeaturesData,
        bool loadSampleIds,
        TVector<TColumn>* columnsDescription,
        TMaybe<size_t>* sampleIdColumnIdx
    );

    // IDataProviderSampler::SampleByIndices implementation for formats with files with lines
    TDataProviderPtr LinesFileSampleByIndices(const TDataProviderSampleParams& params, TConstArrayRef<ui32> indices);

    TDataProviderPtr DataProviderSamplerReorderByIndices(
        const TDataProviderSampleParams& params,
        TDataProviderPtr dataProvider,
        TConstArrayRef<ui32> indices
    );

    TDataProviderPtr DataProviderSamplerReorderBySampleIds(
        const TDataProviderSampleParams& params,
        TDataProviderPtr dataProvider,
        TConstArrayRef<TString> sampleIds
    );
}
