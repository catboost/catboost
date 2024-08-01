#include "sampler.h"

#include "load_data.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/system/compiler.h>


using namespace NCB;


void NCB::ReadCDForSampler(
    const TPathWithScheme& cdPath,
    bool onlyFeaturesData,
    bool loadSampleIds,
    TVector<TColumn>* columnsDescription,
    TMaybe<size_t>* sampleIdColumnIdx
) {
    if (!cdPath.Inited()) {
        return;
    }
    *columnsDescription = ReadCD(cdPath, TCdParserDefaults(EColumn::Num));
    for (auto i : xrange(columnsDescription->size())) {
        auto& column = (*columnsDescription)[i];
        if (column.Type == EColumn::SampleId) {
            CB_ENSURE(!(*sampleIdColumnIdx).Defined(), "Duplicate SampleId columns in CD file");
            (*sampleIdColumnIdx) = i;
            if (onlyFeaturesData && !loadSampleIds) {
                column.Type = EColumn::Auxiliary;
            }
        } else if (onlyFeaturesData && !IsFactorColumn(column.Type)) {
            column.Type = EColumn::Auxiliary;
        }
    }
}


static TDataProviderPtr ReadDatasetForSampler(
    const TDataProviderSampleParams& params,
    bool loadSampleIds,
    THolder<ILineDataReader>&& lineDataReader
) {
    const auto& datasetReadingParams = params.DatasetReadingParams;

    TVector<TColumn> columnsDescription;
    TMaybe<size_t> sampleIdColumnIdx;

    ReadCDForSampler(
        datasetReadingParams.ColumnarPoolFormatParams.CdFilePath,
        params.OnlyFeaturesData,
        loadSampleIds,
        &columnsDescription,
        &sampleIdColumnIdx
    );

    TVector<NJson::TJsonValue> classLabels = datasetReadingParams.ClassLabels;

    return ReadDataset(
        std::move(lineDataReader),
        /*pairsFilePath*/ TPathWithScheme(),
        /*graphFilePath*/ TPathWithScheme(),
        /*groupWeightsFilePath*/ TPathWithScheme(),
        /*timestampsFilePath*/ TPathWithScheme(),
        /*baselineFilePath*/ TPathWithScheme(),
        datasetReadingParams.FeatureNamesPath,
        datasetReadingParams.PoolMetaInfoPath,
        datasetReadingParams.ColumnarPoolFormatParams.DsvFormat,
        columnsDescription,
        /*ignoredFeatures*/ {},  //TODO(akhropov): get unused features from the model
        EObjectsOrder::Ordered,
        loadSampleIds,
        /*forceUnitAutoPairWeights*/ false,
        &classLabels,
        params.LocalExecutor
    );
}

void DegroupDataset(TDataProvider* dataset) {
    auto& metaInfo = dataset->MetaInfo;
    metaInfo.HasGroupId = false;
    metaInfo.HasGroupWeight = false;
    metaInfo.HasSubgroupIds = false;

    if (metaInfo.ColumnsInfo) {
        for (auto& column : metaInfo.ColumnsInfo->Columns) {
            switch (column.Type) {
                case EColumn::GroupId:
                case EColumn::SubgroupId:
                case EColumn::GroupWeight:
                    column.Type = EColumn::Auxiliary;
                default:
                    break;
            }
        }
    }

    auto objectCount = dataset->GetObjectCount();
    dataset->ObjectsGrouping = MakeIntrusive<TObjectsGrouping>(objectCount);
}


TDataProviderPtr NCB::DataProviderSamplerReorderByIndices(
    const TDataProviderSampleParams& params,
    TDataProviderPtr dataProvider,
    TConstArrayRef<ui32> indices
) {
    // resort dataset according to original indices
    TVector<ui32> currentOrder(indices.size());
    Iota(currentOrder.begin(), currentOrder.end(), ui32(0));
    Sort(currentOrder, [&](ui32 l, ui32 r) { return indices[l] < indices[r]; });
    TVector<ui32> finalOrder(indices.size());
    for (size_t i : xrange(indices.size())) {
        finalOrder[currentOrder[i]] = i;
    }

    if (params.OnlyFeaturesData) {
        DegroupDataset(dataProvider.Get());
    }

    return dataProvider->GetSubset(
        GetGroupingSubsetFromObjectsSubset(
            dataProvider->ObjectsGrouping,
            finalOrder,
            EObjectsOrder::RandomShuffled
        ),
        params.CpuUsedRamLimit,
        params.LocalExecutor
    );
}


TDataProviderPtr NCB::LinesFileSampleByIndices(const TDataProviderSampleParams& params, TConstArrayRef<ui32> indices) {
    TVector<ui64> sortedIndices(indices.begin(), indices.end());
    Sort(sortedIndices);

    const auto& datasetReadingParams = params.DatasetReadingParams;

    auto indexedDataReader = MakeHolder<TIndexedSubsetLineDataReader>(
        GetLineDataReader(
            datasetReadingParams.PoolPath,
            datasetReadingParams.ColumnarPoolFormatParams.DsvFormat,
            /*keepInOrder*/ true
        ),
        std::move(sortedIndices)
    );

    return DataProviderSamplerReorderByIndices(
        params,
        ReadDatasetForSampler(params, /*loadSampleIds*/false, std::move(indexedDataReader)),
        indices
    );
}

TDataProviderPtr NCB::DataProviderSamplerReorderBySampleIds(
    const TDataProviderSampleParams& params,
    TDataProviderPtr dataProvider,
    TConstArrayRef<TString> sampleIds
) {
    const ui32 objectCount = dataProvider->GetObjectCount();

    CB_ENSURE_INTERNAL(
        sampleIds.size() == (size_t)objectCount,
        "sampled dataset must have the sample number of objects as sampleIds array"
    );

    // resort dataset according to original sampleIds
    THashMap<TString, TVector<ui32>> sampleIdToIdx;

    for (ui32 i : xrange(objectCount)) {
        sampleIdToIdx[sampleIds[i]].push_back(i);
    }

    TConstArrayRef<TString> datasetSampleIds = dataProvider->ObjectsData->GetSampleIds().GetRef();

    TVector<ui32> finalOrder(sampleIds.size());

    for (ui32 i : xrange(objectCount)) {
        auto dstIt = sampleIdToIdx.find(datasetSampleIds[i]);
        CB_ENSURE_INTERNAL(dstIt != sampleIdToIdx.end(), "dataset sampleId not found in sampleIds");
        auto& mappedIndices = dstIt->second;
        CB_ENSURE_INTERNAL(!mappedIndices.empty(), "empty list of mapped indices");
        finalOrder[mappedIndices.back()] = i;
        mappedIndices.pop_back();
    }

    if (params.OnlyFeaturesData) {
        DegroupDataset(dataProvider.Get());
    }

    return dataProvider->GetSubset(
        GetGroupingSubsetFromObjectsSubset(
            dataProvider->ObjectsGrouping,
            finalOrder,
            EObjectsOrder::RandomShuffled
        ),
        params.CpuUsedRamLimit,
        params.LocalExecutor
    );
}
