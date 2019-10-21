#pragma once

#include <catboost/libs/data/meta_info.h>
#include <catboost/private/libs/options/load_options.h>

#include <library/json/writer/json_value.h>

#include <util/generic/fwd.h>
#include <util/generic/map.h>

TMap<TString, int> ParseMonotonicConstraintsFromString(const TString& monotoneConstraints);
void ConvertFeatureNamesToIndicesInMonotoneConstraints(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* catBoostJsonOptions);
void ConvertFeatureNamesToIndicesInMonotoneConstraints(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions);
void ConvertMonotoneConstraintsToCanonicalFormat(NJson::TJsonValue* catBoostJsonOptions);
