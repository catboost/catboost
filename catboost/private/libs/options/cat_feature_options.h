#pragma once

#include "binarization_options.h"
#include "unimplemented_aware_option.h"

#include <catboost/private/libs/ctr_description/ctr_type.h>

#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>

#include <functional>


namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    using TPrior = TVector<float>;

    NJson::TJsonValue ParseCtrDescription(TStringBuf description);
    NJson::TJsonValue ParseCtrDescriptions(TStringBuf description);
    std::pair<ui32, NJson::TJsonValue> ParsePerFeatureCtrDescription(TStringBuf description);
    NJson::TJsonValue ParsePerFeatureCtrs(TStringBuf description);
    TString BuildCtrOptionsDescription(const NJson::TJsonValue& options);


    TVector<TPrior> GetDefaultPriors(ECtrType ctrType);

    struct TCtrDescription {
        TCtrDescription();
        explicit TCtrDescription(ECtrType type);
        TCtrDescription(ECtrType type, TVector<TPrior> priors);
        TCtrDescription(ECtrType type,
            TVector<TPrior> priors,
            TBinarizationOptions ctrBinarization);
        explicit TCtrDescription(
            ECtrType type,
            TVector<TPrior> priors,
            TBinarizationOptions ctrBinarization,
            TBinarizationOptions targetBinarization);

        void SetPriors(const TVector<TPrior>& priors);

        bool ArePriorsSet() const;

        void Load(const NJson::TJsonValue& options);
        void Save(NJson::TJsonValue* options) const;

        bool operator==(const TCtrDescription& rhs) const;
        bool operator!=(const TCtrDescription& rhs) const;

        const TVector<TPrior>& GetPriors() const;
        const TBinarizationOptions& GetCtrBinarization() const;

        TOption<ECtrType> Type;
        TOption<TVector<TPrior>> Priors;
        TOption<TBinarizationOptions> CtrBinarization;
        TOption<TBinarizationOptions> TargetBinarization;
        TOption<EPriorEstimation> PriorEstimation;

    private:
        void DisableRedundantFields();
    };

    class TCatFeatureParams {
    public:
        explicit TCatFeatureParams(ETaskType taskType);

        void Load(const NJson::TJsonValue& options);
        void Save(NJson::TJsonValue* options) const;

        bool operator==(const TCatFeatureParams& rhs) const;
        bool operator!=(const TCatFeatureParams& rhs) const;

        void Validate() const;

        void AddSimpleCtrDescription(const TCtrDescription& description);
        void AddTreeCtrDescription(const TCtrDescription& description);

        void ForEachCtrDescription(std::function<void(TCtrDescription*)>&& f);
        void ForEachCtrDescription(std::function<void(const TCtrDescription&)>&& f) const;

        TOption<TVector<TCtrDescription>> SimpleCtrs;
        TOption<TVector<TCtrDescription>> CombinationCtrs;
        TOption<TMap<ui32, TVector<TCtrDescription>>> PerFeatureCtrs;

        TOption<TBinarizationOptions> TargetBinarization;

        TOption<ui32> MaxTensorComplexity;
        TOption<ui32> OneHotMaxSize;
        ui32 OneHotMaxSizeLimit;
        TOption<ECounterCalc> CounterCalcMethod;

        TCpuOnlyOption<bool> StoreAllSimpleCtrs;
        TCpuOnlyOption<ui64> CtrLeafCountLimit;

        TGpuOnlyOption<ECtrHistoryUnit> CtrHistoryUnit;
    };

    bool CtrsNeedTargetData(const TCatFeatureParams& catFeatureParams);
}
