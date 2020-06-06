#pragma once

#include <catboost/libs/model/model.h>

#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/array_ref.h>
#include <util/system/hp_timer.h>

class IPerftestModule {
public:
    enum class EPerftestModuleDataLayout {
        ObjectsFirst,
        FeaturesFirst
    };

    /// we will dynamically select module with highest priority as baseline
    virtual int GetComparisonPriority(EPerftestModuleDataLayout layout) const = 0;
    virtual bool SupportsLayout(EPerftestModuleDataLayout layout) const = 0;
    virtual double Do(EPerftestModuleDataLayout layout, TConstArrayRef<TConstArrayRef<float>> features) = 0;
    virtual TString GetName(TMaybe<EPerftestModuleDataLayout> = Nothing()) const = 0;

    virtual ~IPerftestModule() = default;
};

class TBasePerftestModule : public IPerftestModule {
protected:
    THPTimer Timer;
};

using TPerftestModuleFactory = NObjectFactory::TParametrizedObjectFactory<IPerftestModule, TString, const TFullModel&>;
