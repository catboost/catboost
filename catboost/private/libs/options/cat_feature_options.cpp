#include "cat_feature_options.h"
#include "json_helper.h"
#include "restrictions.h"

#include <util/charset/utf8.h>
#include <util/generic/maybe.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/string/strip.h>


namespace {
    struct TCtrParam {
        TString Name;
        TString Value;
    };
}

template <>
inline TCtrParam FromString<TCtrParam>(const TStringBuf& paramBuf) {
    TStringBuf buf = paramBuf;
    TCtrParam param;
    GetNext<TString>(buf, '=', param.Name);
    GetNext<TString>(buf, '=', param.Value);
    return param;
}

static TVector<float> ParsePriors(const TStringBuf priors) {
    TVector<float> result;
    for (const auto& t : StringSplitter(priors).Split('/')) {
        const auto entry = FromString<float>(t.Token());
        result.push_back(entry);
    }
    return result;
}

NJson::TJsonValue NCatboostOptions::ParseCtrDescription(TStringBuf ctrStringDescription) {
    ECtrType type;
    GetNext<ECtrType>(ctrStringDescription, ':', type);

    TSet<TString> seenParams;
    TMaybe<TCtrParam> param;
    GetNext<TCtrParam>(ctrStringDescription, ':', param);
    NJson::TJsonValue ctrJson;
    ctrJson["ctr_type"] = ToString(type);

    TVector<NJson::TJsonValue> priors;
    while (param.Defined()) {
        auto name = ToLowerUTF8(param->Name);
        if (name == "targetbordertype") {
            CB_ENSURE(seenParams.count(name) == 0, "Duplicate param: " << param->Name);
            CB_ENSURE(type != ECtrType::Counter && type != ECtrType::FeatureFreq, "Target borders options are unsupported for counter ctr");
            ctrJson["target_binarization"]["border_type"] = param->Value;
        } else if (name == "targetbordercount") {
            CB_ENSURE(type != ECtrType::Counter && type != ECtrType::FeatureFreq, "Target borders options are unsupported for counter ctr");
            CB_ENSURE(seenParams.count(name) == 0, "Duplicate param: " << param->Name);
            ctrJson["target_binarization"]["border_count"] = FromString<ui32>(param->Value);
        } else if (name == "ctrbordertype") {
            CB_ENSURE(seenParams.count(name) == 0, "Duplicate param: " << param->Name);
            ctrJson["ctr_binarization"]["border_type"] = param->Value;
        } else if (name == "ctrbordercount") {
            CB_ENSURE(seenParams.count(name) == 0, "Duplicate param: " << param->Name);
            ctrJson["ctr_binarization"]["border_count"] = FromString<ui32>(param->Value);
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
            ythrow TCatBoostException() << "Unknown ctr param name: " << param->Name;
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

NJson::TJsonValue
NCatboostOptions::ParseCtrDescriptions(const TStringBuf description) {
    NJson::TJsonValue ctrs(NJson::JSON_ARRAY);
    for (const auto oneCtrConfig : StringSplitter(description).Split(',').SkipEmpty()) {
        ctrs.AppendValue(ParseCtrDescription(oneCtrConfig.Token()));
    }
    CB_ENSURE(!ctrs.GetArray().empty(), "Empty ctr description " << description);
    return ctrs;
}

std::pair<ui32, NJson::TJsonValue>
NCatboostOptions::ParsePerFeatureCtrDescription(TStringBuf ctrStringDescription) {
    ui32 featureId;
    GetNext<ui32>(ctrStringDescription, ':', featureId);
    std::pair<ui32, NJson::TJsonValue> perFeatureCtr;
    perFeatureCtr.first = featureId;
    perFeatureCtr.second = ParseCtrDescriptions(ctrStringDescription);
    return perFeatureCtr;
}

NJson::TJsonValue NCatboostOptions::ParsePerFeatureCtrs(const TStringBuf description) {
    NJson::TJsonValue perFeaturesCtrsMap(NJson::JSON_MAP);

    for (const auto& onePerFeatureCtrConfig : StringSplitter(description).Split(';')) {
        auto perFeatureCtr = ParsePerFeatureCtrDescription(onePerFeatureCtrConfig.Token());
        perFeaturesCtrsMap[ToString(perFeatureCtr.first)] = perFeatureCtr.second;
    }
    return perFeaturesCtrsMap;
}

TVector<NCatboostOptions::TPrior> NCatboostOptions::GetDefaultPriors(const ECtrType ctrType) {
    switch (ctrType) {
        case ECtrType::Borders:
        case ECtrType::Buckets:
        case ECtrType::BinarizedTargetMeanValue: {
            return {{0, 1},
                    {0.5, 1},
                    {1, 1}};
        }
        case ECtrType::FeatureFreq:
        case ECtrType::Counter: {
            return {{0.0, 1}};
        }
        case ECtrType::FloatTargetMeanValue: {
            return {{0, 1}};
        }
        default: {
            ythrow TCatBoostException() << "Unknown ctr type " << ctrType;
        }
    }
}

NCatboostOptions::TCtrDescription::TCtrDescription(
    const ECtrType type,
    TVector<TPrior> priors,
    TBinarizationOptions ctrBinarization,
    TBinarizationOptions targetBinarization)
    : Type("ctr_type", type)
    , Priors("priors", std::move(priors))
    , CtrBinarization("ctr_binarization", std::move(ctrBinarization))
    , TargetBinarization("target_binarization", std::move(targetBinarization))
    , PriorEstimation("prior_estimation", EPriorEstimation::No)
{
    DisableRedundantFields();
}

NCatboostOptions::TCtrDescription::TCtrDescription(
    const ECtrType type,
    TVector<TPrior> priors,
    TBinarizationOptions ctrBinarization)
    : TCtrDescription(
        type,
        std::move(priors),
        std::move(ctrBinarization),
        TBinarizationOptions(EBorderSelectionType::MinEntropy, 1)) {
}

NCatboostOptions::TCtrDescription::TCtrDescription(const ECtrType type, TVector<TPrior> priors)
    : TCtrDescription(
        type,
        std::move(priors),
        TBinarizationOptions(EBorderSelectionType::Uniform, 15)) {
}

NCatboostOptions::TCtrDescription::TCtrDescription(const ECtrType type)
    : TCtrDescription(type, {}) {
}
NCatboostOptions::TCtrDescription::TCtrDescription()
    : TCtrDescription(ECtrType::Borders, {}) {
}

void NCatboostOptions::TCtrDescription::SetPriors(const TVector<TPrior>& priors) {
    return Priors.Set(priors);
}

bool NCatboostOptions::TCtrDescription::ArePriorsSet() const {
    return Priors.IsSet();
}


void NCatboostOptions::TCtrDescription::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &Type, &Priors, &CtrBinarization, &TargetBinarization, &PriorEstimation);
    DisableRedundantFields();
}

void NCatboostOptions::TCtrDescription::Save(NJson::TJsonValue* options) const {
    SaveFields(options, Type, Priors, CtrBinarization, TargetBinarization, PriorEstimation);
}

bool NCatboostOptions::TCtrDescription::operator==(const TCtrDescription& rhs) const {
    return std::tie(Type, Priors, CtrBinarization, TargetBinarization, PriorEstimation) ==
        std::tie(rhs.Type, rhs.Priors, rhs.CtrBinarization, rhs.TargetBinarization, rhs.PriorEstimation);
}

bool NCatboostOptions::TCtrDescription::operator!=(const TCtrDescription& rhs) const {
    return !(rhs == *this);
}

const TVector<NCatboostOptions::TPrior>& NCatboostOptions::TCtrDescription::GetPriors() const {
    return Priors.Get();
}

const NCatboostOptions::TBinarizationOptions& NCatboostOptions::TCtrDescription::GetCtrBinarization() const {
    return CtrBinarization.Get();
}

void NCatboostOptions::TCtrDescription::DisableRedundantFields() {
    const ECtrType ctrType = Type;
    if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
        TargetBinarization.SetDisabledFlag(true);
    } else {
        TargetBinarization->DisableNanModeOption();
    }
    TargetBinarization->DisableMaxSubsetSizeForBuildBordersOption();
    CtrBinarization->DisableNanModeOption();
    CtrBinarization->DisableMaxSubsetSizeForBuildBordersOption();
}

NCatboostOptions::TCatFeatureParams::TCatFeatureParams(ETaskType taskType)
    : SimpleCtrs("simple_ctrs", TVector<TCtrDescription>())
    , CombinationCtrs("combinations_ctrs", TVector<TCtrDescription>())
    , PerFeatureCtrs("per_feature_ctrs", TMap<ui32, TVector<TCtrDescription>>())
    , TargetBinarization("target_binarization", TBinarizationOptions(EBorderSelectionType::MinEntropy, 1))
    , MaxTensorComplexity("max_ctr_complexity", 4)
    , OneHotMaxSize("one_hot_max_size", 2)
    , OneHotMaxSizeLimit(taskType == ETaskType::CPU ? GetMaxBinCount() : 256) // there is still limit for OneHot on GPU
    , CounterCalcMethod("counter_calc_method", ECounterCalc::SkipTest)
    , StoreAllSimpleCtrs("store_all_simple_ctr", false, taskType)
    , CtrLeafCountLimit("ctr_leaf_count_limit", Max<ui64>(), taskType)
    , CtrHistoryUnit("ctr_history_unit", ECtrHistoryUnit::Sample, taskType) {
    TargetBinarization.Get().DisableNanModeOption();
    TargetBinarization.Get().DisableMaxSubsetSizeForBuildBordersOption();
}

void NCatboostOptions::TCatFeatureParams::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options,
            &SimpleCtrs, &CombinationCtrs, &PerFeatureCtrs, &TargetBinarization, &MaxTensorComplexity, &OneHotMaxSize, &CounterCalcMethod,
            &StoreAllSimpleCtrs, &CtrLeafCountLimit, &CtrHistoryUnit);
    Validate();
}

void NCatboostOptions::TCatFeatureParams::Save(NJson::TJsonValue* options) const {
    SaveFields(options,
            SimpleCtrs, CombinationCtrs, PerFeatureCtrs, TargetBinarization, MaxTensorComplexity, OneHotMaxSize, CounterCalcMethod,
            StoreAllSimpleCtrs, CtrLeafCountLimit, CtrHistoryUnit);
}

bool NCatboostOptions::TCatFeatureParams::operator==(const TCatFeatureParams& rhs) const {
    return std::tie(SimpleCtrs, CombinationCtrs, PerFeatureCtrs, TargetBinarization, MaxTensorComplexity, OneHotMaxSize, CounterCalcMethod,
            StoreAllSimpleCtrs, CtrLeafCountLimit, CtrHistoryUnit) ==
        std::tie(rhs.SimpleCtrs, rhs.CombinationCtrs, rhs.PerFeatureCtrs, rhs.TargetBinarization, rhs.MaxTensorComplexity, rhs.OneHotMaxSize,
                rhs.CounterCalcMethod, rhs.StoreAllSimpleCtrs, rhs.CtrLeafCountLimit, rhs.CtrHistoryUnit);
}

bool NCatboostOptions::TCatFeatureParams::operator!=(const TCatFeatureParams& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TCatFeatureParams::Validate() const {
    CB_ENSURE(OneHotMaxSize.Get() <= OneHotMaxSizeLimit,
            "Error in one_hot_max_size: maximum value of one-hot-encoding is " << OneHotMaxSizeLimit);
    const ui32 ctrComplexityLimit = GetMaxTreeDepth();
    CB_ENSURE(MaxTensorComplexity.Get() < ctrComplexityLimit,
            "Error: max ctr complexity should be less than " << ctrComplexityLimit);
    if (!CtrLeafCountLimit.IsUnimplementedForCurrentTask()) {
        CB_ENSURE(CtrLeafCountLimit.Get() > 0,
                "Error: ctr_leaf_count_limit must be positive");
    }
}

void NCatboostOptions::TCatFeatureParams::AddSimpleCtrDescription(const TCtrDescription& description) {
    SimpleCtrs->push_back(description);
}

void NCatboostOptions::TCatFeatureParams::AddTreeCtrDescription(const TCtrDescription& description) {
    CombinationCtrs->push_back(description);
}


void NCatboostOptions::TCatFeatureParams::ForEachCtrDescription(
    std::function<void(NCatboostOptions::TCtrDescription*)>&& f) {

    for (auto& ctrDescription : SimpleCtrs.Get()) {
        f(&ctrDescription);
    }
    for (auto& ctrDescription : CombinationCtrs.Get()) {
        f(&ctrDescription);
    }
    for (auto& [id, perFeatureCtr] : PerFeatureCtrs.Get()) {
        for (auto& ctrDescription : perFeatureCtr) {
            f(&ctrDescription);
        }
    }
}

void NCatboostOptions::TCatFeatureParams::ForEachCtrDescription(
    std::function<void(const NCatboostOptions::TCtrDescription&)>&& f) const {

    for (const auto& ctrDescription : SimpleCtrs.Get()) {
        f(ctrDescription);
    }
    for (const auto& ctrDescription : CombinationCtrs.Get()) {
        f(ctrDescription);
    }
    for (const auto&  [id, perFeatureCtr] : PerFeatureCtrs.Get()) {
        for (const auto&  ctrDescription : perFeatureCtr) {
            f(ctrDescription);
        }
    }
}

bool NCatboostOptions::CtrsNeedTargetData(const NCatboostOptions::TCatFeatureParams& catFeatureParams) {
    bool ctrsNeedTargetData = false;

    catFeatureParams.ForEachCtrDescription(
        [&] (const NCatboostOptions::TCtrDescription& ctrDescription) {
            if (NeedTarget(ctrDescription.Type)) {
                ctrsNeedTargetData = true;
            }
        }
    );

    return ctrsNeedTargetData;
}

TString NCatboostOptions::BuildCtrOptionsDescription(const NJson::TJsonValue& options)
{
    TString ctrTypeStringConcat = ToString(options["ctr_type"]);
    ctrTypeStringConcat = StripString(ctrTypeStringConcat, EqualsStripAdapter('"'));

    if (options["ctr_binarization"].Has("border_count")) {
        auto ctrBorderCount = ToString(options["ctr_binarization"]["border_count"]);
        ctrTypeStringConcat = ctrTypeStringConcat + ":CtrBorderCount=" + ctrBorderCount;
    }

    if (options["ctr_binarization"].Has("border_type")) {
        auto ctrBorderTypeStripped = StripString(ToString(options["ctr_binarization"]["border_type"]),
                                                 EqualsStripAdapter('"'));
        ctrTypeStringConcat =
                ctrTypeStringConcat + ":CtrBorderType=" + ctrBorderTypeStripped;
    }

    if (options["target_binarization"].Has("border_count")) {
        auto targetBorderCount = ToString(options["target_binarization"]["border_count"]);
        ctrTypeStringConcat = ctrTypeStringConcat + ":TargetBorderCount=" + targetBorderCount;
    }

    if (options["target_binarization"].Has("border_type")) {
        auto targetBorderTypeStripped = StripString(ToString(options["target_binarization"]["border_type"]),
                                                    EqualsStripAdapter('"'));
        ctrTypeStringConcat =
                ctrTypeStringConcat + ":TargetBorderType=" + targetBorderTypeStripped;
    }

    const NJson::TJsonValue& priorDescriptions = options["priors"];
    if (priorDescriptions.IsArray()) {
        for (const auto& prior : priorDescriptions.GetArraySafe()) {
            auto numerator = ToString(prior[0]);

            // for GPU
            if (prior.Has(1)) {
                numerator  = numerator + "/" + ToString(prior[1]);
            }
            ctrTypeStringConcat = ctrTypeStringConcat + ":Prior=" + numerator;
        }
    }
    return ctrTypeStringConcat;
}

