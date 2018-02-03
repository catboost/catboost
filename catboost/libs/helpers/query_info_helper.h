#pragma once

#include <catboost/libs/data_types/query.h>

#include <util/generic/vector.h>

void UpdateQueriesInfo(const TVector<ui32>& queriesId, int begin, int end, TVector<TQueryInfo>* queryInfo);

TVector<TQueryEndInfo> GetQueryEndInfo(const TVector<TQueryInfo>& queriesInfo, int learnSampleCount);

