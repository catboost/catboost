#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/system/types.h>
#include <util/ysaveload.h>


struct TShapValue {
    int Feature = -1;
    TVector<double> Value;

    TShapValue() = default;

    TShapValue(int feature, int approxDimension)
        : Feature(feature)
        , Value(approxDimension)
    {
    }

    Y_SAVELOAD_DEFINE(Feature, Value);
};

struct TShapPreparedTrees {
    TVector<TVector<TVector<TShapValue>>> ShapValuesByLeafForAllTrees;
    TVector<TVector<double>> MeanValuesForAllTrees;

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

// returned: ShapValues[documentIdx][dimenesion][feature]
TVector<TVector<TVector<double>>> CalcShapValuesMulti(
    const TFullModel& model,
    const TPool& pool,
    NPar::TLocalExecutor* localExecutor,
    int logPeriod = 0
);

// returned: ShapValues[documentIdx][feature]
TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const TPool& pool,
    NPar::TLocalExecutor* localExecutor,
    int logPeriod = 0
);

// outputs for each document in order for each dimension in order an array of feature contributions
void CalcAndOutputShapValues(
    const TFullModel& model,
    const TPool& pool,
    const TString& outputPath,
    int threadCount,
    int logPeriod = 0
);
