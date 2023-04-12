%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/pairs.h>
%}

%catches(std::exception) TPairsDataBuilder::Add(i64 groupIdx, i32 winnerIdxInGroup, i32 loserIdxInGroup);
%catches(std::exception) TPairsDataBuilder::Add(
    i64 groupIdx, 
    i32 winnerIdxInGroup, 
    i32 loserIdxInGroup, 
    float weight
);

%catches(yexception) TPairsDataBuilder::AddToResult(NCB::IQuantizedFeaturesDataVisitor* visitor);

%catches(yexception) SavePairsInGroupedDsvFormat(
    const NCB::TDataProviderPtr& dataProvider,
    const TString& outputFile
);

%include "pairs.h"
