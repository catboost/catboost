%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/model_application.h>
%}

%include "catboost_enums.i"
%include "data_provider.i"
%include "model.i"
%include "tvector.i"


%catches(yexception) TApplyResultIterator::TApplyResultIterator(
    const TFullModel& model,
    NCB::TObjectsDataProviderPtr objectsDataProvider,
    EPredictionType predictionType,
    NPar::TLocalExecutor* localExecutor
);

%catches(yexception) CheckModelAndDatasetCompatibility(
    const TFullModel& model,
    const NCB::TQuantizedFeaturesInfo& datasetQuantizedFeaturesInfo
);

%catches(yexception) CreateQuantizedFeaturesInfoForModelApplication(
    const TFullModel& model,
    const NCB::TFeaturesLayout& datasetFeaturesLayout
);

%include "model_application.h"
