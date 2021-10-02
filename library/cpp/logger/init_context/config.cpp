#include "config.h"

TLogBackendCreatorInitContextConfig::TLogBackendCreatorInitContextConfig(const NConfig::TConfig& config)
    : Config(config)
{}

bool TLogBackendCreatorInitContextConfig::GetValue(TStringBuf name, TString& var) const {
    if (Config.Has(name)) {
        var = Config[name].Get<TString>();
        return true;
    }
    return false;
}

TVector<THolder<ILogBackendCreator::IInitContext>> TLogBackendCreatorInitContextConfig::GetChildren(TStringBuf name) const {
    TVector<THolder<IInitContext>> result;
    const NConfig::TConfig& child = Config[name];
    if (child.IsA<NConfig::TArray>()) {
        for (const auto& i: child.Get<NConfig::TArray>()) {
            result.emplace_back(MakeHolder<TLogBackendCreatorInitContextConfig>(i));
        }
    } else if (!child.IsNull()) {
        result.emplace_back(MakeHolder<TLogBackendCreatorInitContextConfig>(child));
    }
    return result;
}
