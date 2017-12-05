#include "cat_feature_options.h"
#include <util/string/cast.h>
#include <util/charset/utf8.h>
#include <util/string/iterator.h>

struct TCtrParam {
    TString Name;
    TString Value;
};

template <>
inline TCtrParam FromString<TCtrParam>(const TStringBuf& paramBuf) {
    TStringBuf buf = paramBuf;
    TCtrParam param;
    GetNext<TString>(buf, '=', param.Name);
    GetNext<TString>(buf, '=', param.Value);
    return param;
}

inline TVector<float> ParsePriors(const TString& priors) {
    TVector<float> result;
    for (const auto& t : StringSplitter(priors).Split('/')) {
        const auto entry = FromString<float>(t.Token());
        result.push_back(entry);
    }
    return result;
}

namespace NCatboostOptions {
    NJson::TJsonValue ParseCtrDescription(const TString& description) {
        TStringBuf ctrStringDescription = description;

        ECtrType type;
        GetNext<ECtrType>(ctrStringDescription, ':', type);

        TSet<TString> seenParams;
        TMaybe<TCtrParam> param;
        GetNext<TCtrParam>(ctrStringDescription, ':', param);
        NJson::TJsonValue ctrJson;
        ctrJson["ctr_type"] = ToString<ECtrType>(type);

        TVector<NJson::TJsonValue> priors;
        while (param.Defined()) {
            auto name = ToLowerUTF8(param->Name);
            if (name == "targetbordertype") {
                CB_ENSURE(seenParams.count(name) == 0, "Duplicate param: " << param->Name);
                CB_ENSURE(type != ECtrType::Counter && type != ECtrType::FeatureFreq, "Target borders options are unsupported for counter ctr");
                ctrJson["target_borders"]["border_type"] = param->Value;
            } else if (name == "targetbordercount") {
                CB_ENSURE(type != ECtrType::Counter && type != ECtrType::FeatureFreq, "Target borders options are unsupported for counter ctr");
                CB_ENSURE(seenParams.count(name) == 0, "Duplicate param: " << param->Name);
                ctrJson["target_borders"]["border_count"] = FromString<ui32>(param->Value);
            } else if (name == "ctrbordertype") {
                CB_ENSURE(seenParams.count(name) == 0, "Duplicate param: " << param->Name);
                ctrJson["ctr_borders"]["border_type"] = param->Value;
            } else if (name == "ctrbordercount") {
                CB_ENSURE(seenParams.count(name) == 0, "Duplicate param: " << param->Name);
                ctrJson["ctr_borders"]["border_count"] = FromString<ui32>(param->Value);
            } else if (name == "prior") {
                auto priorParams = ParsePriors(param->Value);
                NJson::TJsonValue jsonPrior(NJson::JSON_ARRAY);
                for (const auto& entry : priorParams) {
                    jsonPrior.AppendValue(entry);
                }
                priors.push_back(jsonPrior);
            } else if (name == "priorestimation") {
                CB_ENSURE(seenParams.count(name) == 0, "Duplicate param: " << param->Name);
                ctrJson["prior_estimation"] = param->Value;
            } else {
                ythrow TCatboostException() << "Unknown ctr param name: " << param->Name;
            }
            seenParams.insert(name);
            GetNext<TCtrParam>(ctrStringDescription, ':', param);
        }
        if (priors.size()) {
            for (const auto& prior : priors) {
                ctrJson["priors"].AppendValue(prior);
            }
        }
        return ctrJson;
    }

    std::pair<ui32, NJson::TJsonValue> ParsePerFeatureCtrDescription(const TString& description) {
        TStringBuf ctrStringDescription = description;
        ui32 featureId;
        GetNext<ui32>(ctrStringDescription, ':', featureId);
        std::pair<ui32, NJson::TJsonValue> perFeatureCtr;
        perFeatureCtr.first = featureId;
        perFeatureCtr.second = ParseCtrDescriptions(TString(ctrStringDescription));
        return perFeatureCtr;
    }
}
