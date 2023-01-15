#include "ctrs.h"

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_writer.h>

#include <util/generic/cast.h>
#include <util/stream/str.h>


void NCB::TPrecomputedOnlineCtrMetaData::Append(TPrecomputedOnlineCtrMetaData& rhs) {
    ui32 featureIdxOffset = SafeIntegerCast<ui32>(OnlineCtrIdxToFeatureIdx.size());
    for (const auto& [onlineCtrIdx, featureIdx]: rhs.OnlineCtrIdxToFeatureIdx) {
        CB_ENSURE(
            OnlineCtrIdxToFeatureIdx.emplace(onlineCtrIdx, featureIdxOffset + featureIdx).second,
            "Duplicate onlineCtrIdx while appending to TPrecomputedOnlineCtrMetaData::OnlineCtrIdxToFeatureIdx"
        );
    }
    for (const auto& [catFeatureIdx, valuesCounts] : rhs.ValuesCounts) {
        CB_ENSURE(
            ValuesCounts.emplace(catFeatureIdx, valuesCounts).second,
            "Duplicate catFeatureIdx while appending to TPrecomputedOnlineCtrMetaData::ValuesCounts"
        );
    }
}


TString NCB::TPrecomputedOnlineCtrMetaData::SerializeToJson() const {
    NJson::TJsonMap json;

    NJson::TJsonArray onlineCtrIdxToFeatureIdxJson;
    for (const auto& [onlineCtrIdx, featureIdx] : OnlineCtrIdxToFeatureIdx) {
        NJson::TJsonMap json;
        json.InsertValue("CatFeatureIdx", NJson::TJsonValue(onlineCtrIdx.CatFeatureIdx));
        json.InsertValue("CtrIdx", NJson::TJsonValue(onlineCtrIdx.CtrIdx));
        json.InsertValue("TargetBorderIdx", NJson::TJsonValue(onlineCtrIdx.TargetBorderIdx));
        json.InsertValue("PriorIdx", NJson::TJsonValue(onlineCtrIdx.PriorIdx));
        json.InsertValue("FeatureIdx", NJson::TJsonValue(featureIdx));
        onlineCtrIdxToFeatureIdxJson.AppendValue(std::move(json));
    }
    json.InsertValue("OnlineCtrIdxToFeatureIdx", std::move(onlineCtrIdxToFeatureIdxJson));

    NJson::TJsonArray valuesCountsJson;
    for (const auto& [catFeatureIdx, valuesCounts] : ValuesCounts) {
        NJson::TJsonMap json;
        json.InsertValue("CatFeatureIdx", NJson::TJsonValue(catFeatureIdx));
        json.InsertValue("Count", NJson::TJsonValue(valuesCounts.Count));
        json.InsertValue("CounterCount", NJson::TJsonValue(valuesCounts.CounterCount));
        valuesCountsJson.AppendValue(std::move(json));
    }
    json.InsertValue("ValuesCounts", std::move(valuesCountsJson));

    return NJson::WriteJson(json);
}

NCB::TPrecomputedOnlineCtrMetaData NCB::TPrecomputedOnlineCtrMetaData::DeserializeFromJson(
    const TString& serializedJson
) {
    TStringInput in(serializedJson);
    NJson::TJsonValue json = NJson::ReadJsonTree(&in, /*throwOnError*/ true);

    NCB::TPrecomputedOnlineCtrMetaData result;
    for (const auto& element : json["OnlineCtrIdxToFeatureIdx"].GetArray()) {
        TOnlineCtrIdx onlineCtrIdx;
        onlineCtrIdx.CatFeatureIdx = element["CatFeatureIdx"].GetInteger();
        onlineCtrIdx.CtrIdx = element["CtrIdx"].GetInteger();
        onlineCtrIdx.TargetBorderIdx = element["TargetBorderIdx"].GetInteger();
        onlineCtrIdx.PriorIdx = element["PriorIdx"].GetInteger();
        result.OnlineCtrIdxToFeatureIdx.emplace(std::move(onlineCtrIdx), element["FeatureIdx"].GetInteger());
    }
    for (const auto& element : json["ValuesCounts"].GetArray()) {
        TOnlineCtrUniqValuesCounts valuesCounts;
        valuesCounts.Count = element["Count"].GetInteger();
        valuesCounts.CounterCount = element["CounterCount"].GetInteger();
        result.ValuesCounts.emplace(
            SafeIntegerCast<ui32>(element["CatFeatureIdx"].GetUInteger()),
            std::move(valuesCounts)
        );
    }
    return result;
}
