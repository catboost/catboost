#pragma once

#include <library/cpp/logger/backend_creator.h>
#include <library/cpp/yconf/conf.h>

class TLogBackendCreatorInitContextYConf: public ILogBackendCreator::IInitContext {
public:
    TLogBackendCreatorInitContextYConf(const TYandexConfig::Section& section);
    virtual bool GetValue(TStringBuf name, TString& var) const override;
    virtual TVector<THolder<IInitContext>> GetChildren(TStringBuf name) const override;
private:
    const TYandexConfig::Section& Section;
};
