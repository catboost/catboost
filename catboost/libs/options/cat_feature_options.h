#pragma once

#include "restrictions.h"
#include "binarization_options.h"
#include <catboost/libs/ctr_description/ctr_type.h>
#include <util/generic/vector.h>
#include <util/string/split.h>
#include <util/string/iterator.h>
#include <util/generic/maybe.h>

namespace NCatboostOptions {
    using TPrior = TVector<float>;

    NJson::TJsonValue ParseCtrDescription(const TString& description);

    inline NJson::TJsonValue ParseCtrDescriptions(const TString& description) {
        NJson::TJsonValue ctrs(NJson::JSON_ARRAY);
        for (const auto& oneCtrConfig : StringSplitter(description).Split(',')) {
            ctrs.AppendValue(ParseCtrDescription(TString(oneCtrConfig.Token())));
        }
        return ctrs;
    }
    std::pair<ui32, NJson::TJsonValue> ParsePerFeatureCtrDescription(const TString& description);

    inline NJson::TJsonValue ParsePerFeatureCtrs(const TString& description) {
        NJson::TJsonValue perFeaturesCtrsMap(NJson::JSON_MAP);

        for (const auto& onePerFeatureCtrConfig : StringSplitter(description).Split(';')) {
            auto perFeatureCtr = ParsePerFeatureCtrDescription(TString(onePerFeatureCtrConfig.Token()));
            perFeaturesCtrsMap[ToString<ui32>(perFeatureCtr.first)] = perFeatureCtr.second;
        }
        return perFeaturesCtrsMap;
    }

    inline TVector<TPrior> GetDefaultPriors(ECtrType ctrType) {
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
                ythrow TCatboostException() << "Unknown ctr type " << ctrType;
            }
        }
    }

    struct TCtrDescription {
        explicit TCtrDescription(ECtrType type = ECtrType::Borders,
                                 TVector<TPrior> priors = TVector<TPrior>(),
                                 TBinarizationOptions ctrBinarization = TBinarizationOptions(EBorderSelectionType::Uniform, 15),
                                 TBinarizationOptions targetBinarization = TBinarizationOptions(EBorderSelectionType::MinEntropy, 1))
            : Type("ctr_type", type)
            , Priors("priors", priors)
            , CtrBinarization("ctr_borders", ctrBinarization)
            , TargetBinarization("target_borders", targetBinarization)
            , PriorEstimation("prior_estimation", EPriorEstimation::No) {
            DisableRedundantFields();
        }

        void SetPriors(const TVector<TPrior>& priors) {
            return Priors.Set(priors);
        }

        bool ArePriorsSet() const {
            return Priors.IsSet();
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options,
                        &Type, &Priors, &CtrBinarization, &TargetBinarization, &PriorEstimation);
            DisableRedundantFields();
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, Type, Priors, CtrBinarization, TargetBinarization, PriorEstimation);
        }

        bool operator==(const TCtrDescription& rhs) const {
            return std::tie(Type, Priors, CtrBinarization, TargetBinarization, PriorEstimation) ==
                   std::tie(rhs.Type, rhs.Priors, rhs.CtrBinarization, rhs.TargetBinarization, rhs.PriorEstimation);
        }

        bool operator!=(const TCtrDescription& rhs) const {
            return !(rhs == *this);
        }

        const TVector<TPrior>& GetPriors() const {
            return Priors.Get();
        }

        const TBinarizationOptions& GetCtrBinarization() const {
            return CtrBinarization.Get();
        }

        TOption<ECtrType> Type;
        TOption<TVector<TPrior>> Priors;
        TOption<TBinarizationOptions> CtrBinarization;
        TOption<TBinarizationOptions> TargetBinarization;
        TOption<EPriorEstimation> PriorEstimation;

    private:
        void DisableRedundantFields() {
            const ECtrType ctrType = Type;
            if (ctrType == ECtrType::Counter || ctrType == ECtrType::FeatureFreq) {
                TargetBinarization.SetDisabledFlag(true);
            } else {
                TargetBinarization->DisableNanModeOption();
            }
            CtrBinarization->DisableNanModeOption();
        }
    };

    class TCatFeatureParams {
    public:
        explicit TCatFeatureParams(ETaskType taskType)
            : SimpleCtrs("simple_ctrs", TVector<TCtrDescription>())
            , CombinationCtrs("combinations_ctrs", TVector<TCtrDescription>())
            , PerFeatureCtrs("per_feature_ctrs", TMap<ui32, TVector<TCtrDescription>>())
            , MaxTensorComplexity("max_ctr_complexity", 4)
            , OneHotMaxSize("one_hot_max_size", 2)
            , CounterCalcMethod("counter_calc_method", ECounterCalc::Full)
            , StoreAllSimpleCtrs("store_all_simple_ctr", false, taskType)
            , CtrLeafCountLimit("ctr_leaf_count_limit", Max<ui64>(), taskType)
            , TargetBorders("target_borders", TBinarizationOptions(EBorderSelectionType::MinEntropy, 1), taskType) {
            TargetBorders.GetUnchecked().DisableNanModeOption();
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options,
                        &SimpleCtrs, &CombinationCtrs, &PerFeatureCtrs, &MaxTensorComplexity, &OneHotMaxSize, &CounterCalcMethod,
                        &StoreAllSimpleCtrs, &CtrLeafCountLimit, &TargetBorders);
            Validate();
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options,
                       SimpleCtrs, CombinationCtrs, PerFeatureCtrs, MaxTensorComplexity, OneHotMaxSize, CounterCalcMethod,
                       StoreAllSimpleCtrs, CtrLeafCountLimit, TargetBorders);
        }

        bool operator==(const TCatFeatureParams& rhs) const {
            return std::tie(SimpleCtrs, CombinationCtrs, PerFeatureCtrs, MaxTensorComplexity, OneHotMaxSize, CounterCalcMethod,
                            StoreAllSimpleCtrs, CtrLeafCountLimit, TargetBorders) ==
                   std::tie(rhs.SimpleCtrs, rhs.CombinationCtrs, rhs.PerFeatureCtrs, rhs.MaxTensorComplexity, rhs.OneHotMaxSize,
                            rhs.CounterCalcMethod, rhs.StoreAllSimpleCtrs, rhs.CtrLeafCountLimit, rhs.TargetBorders);
        }

        bool operator!=(const TCatFeatureParams& rhs) const {
            return !(rhs == *this);
        }

        void Validate() const {
            CB_ENSURE(OneHotMaxSize.Get() <= GetMaxBinCount(),
                      "Error in one_hot_max_size: maximum value of one-hot-encoding is 255");
            const ui32 ctrComplexityLimit = GetMaxTreeDepth();
            CB_ENSURE(MaxTensorComplexity.Get() < ctrComplexityLimit,
                      "Error: max ctr complexity should be less then " << ctrComplexityLimit);
        }

        void AddSimpleCtrDescription(const TCtrDescription& description) {
            SimpleCtrs->push_back(description);
        }

        void AddTreeCtrDescription(const TCtrDescription& description) {
            CombinationCtrs->push_back(description);
        }

        TOption<TVector<TCtrDescription>> SimpleCtrs;
        TOption<TVector<TCtrDescription>> CombinationCtrs;
        TOption<TMap<ui32, TVector<TCtrDescription>>> PerFeatureCtrs;

        TOption<ui32> MaxTensorComplexity;
        TOption<ui32> OneHotMaxSize;
        TOption<ECounterCalc> CounterCalcMethod;

        TCpuOnlyOption<bool> StoreAllSimpleCtrs;
        TCpuOnlyOption<ui64> CtrLeafCountLimit;

        TGpuOnlyOption<TBinarizationOptions> TargetBorders;
    };

}
