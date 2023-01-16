#pragma once

#include <library/cpp/logger/backend_creator.h>
#include <library/cpp/config/config.h>

class TLogBackendCreatorInitContextConfig : public ILogBackendCreator::IInitContext {
public:
    TLogBackendCreatorInitContextConfig(const NConfig::TConfig& config);
    virtual bool GetValue(TStringBuf name, TString& var) const override;
    virtual TVector<THolder<IInitContext>> GetChildren(TStringBuf name) const override;

private:
    const NConfig::TConfig& Config;
};
