#pragma once

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/model/model.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/system/types.h>
#include <util/ysaveload.h>


struct TShapValue {
    int Feature = -1;
    TVector<double> Value;

public:
    TShapValue() = default;

    TShapValue(int feature, int approxDimension)
        : Feature(feature)
        , Value(approxDimension)
    {
    }

    Y_SAVELOAD_DEFINE(Feature, Value);
};

struct TShapPreparedTrees {
    TVector<TVector<TVector<TShapValue>>> ShapValuesByLeafForAllTrees; // [treeIdx][leafIdx][shapFeature] trees * 2^d * d
    TVector<TVector<double>> MeanValuesForAllTrees;

public:
    TShapPreparedTrees() = default;

    TShapPreparedTrees(
        const TVector<TVector<TVector<TShapValue>>>& shapValuesByLeafForAllTrees,
        const TVector<TVector<double>>& meanValuesForAllTrees
    )
        : ShapValuesByLeafForAllTrees(shapValuesByLeafForAllTrees)
        , MeanValuesForAllTrees(meanValuesForAllTrees)
    {
    }

    Y_SAVELOAD_DEFINE(ShapValuesByLeafForAllTrees, MeanValuesForAllTrees);
};

void CalcShapValuesForDocumentMulti(
    const TObliviousTrees& forest,
    const TShapPreparedTrees& preparedTrees,
    const TVector<ui8>& binarizedFeaturesForBlock,
    int flatFeatureCount,
    size_t documentIdx,
    size_t documentCount,
    TVector<TVector<double>>* shapValues
);

TShapPreparedTrees PrepareTrees(const TFullModel& model, NPar::TLocalExecutor* localExecutor);
TShapPreparedTrees PrepareTrees(
        const TFullModel& model,
        const NCB::TDataProvider* dataset, // can be nullptr if model has LeafWeights
        int logPeriod,
        NPar::TLocalExecutor* localExecutor,
        bool calcInternalValues = false
);

// returned: ShapValues[documentIdx][dimenesion][feature]
TVector<TVector<TVector<double>>> CalcShapValuesMulti(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    int logPeriod,
    NPar::TLocalExecutor* localExecutor
);

// returned: ShapValues[documentIdx][feature]
TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    int logPeriod,
    NPar::TLocalExecutor* localExecutor
);

// outputs for each document in order for each dimension in order an array of feature contributions
void CalcAndOutputShapValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const TString& outputPath,
    int logPeriod,
    NPar::TLocalExecutor* localExecutor
);

void CalcShapValuesInternalForFeature(
        TShapPreparedTrees& preparedTrees,
        const TFullModel& model,
        const NCB::TDataProvider& dataset,
        int logPeriod,
        ui32 start,
        ui32 end,
        ui32 featuresCount,
        const NCB::TRawObjectsDataProvider* rawObjectsData,
        TVector<TVector<TVector<double>>>* shapValues, // [docIdx][featureIdx][dim]
        NPar::TLocalExecutor* localExecutor
);
