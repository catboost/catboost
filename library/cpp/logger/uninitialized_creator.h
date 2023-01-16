#pragma once

#include "backend_creator.h"

class TLogBackendCreatorUninitialized : public ILogBackendCreator {
public:
    void InitCustom(const TString& type, ELogPriority priority, bool threaded);
    virtual bool Init(const IInitContext& ctx) override;
    virtual void ToJson(NJson::TJsonValue& value) const override;

    static TFactory::TRegistrator<TLogBackendCreatorUninitialized> Registrar;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
    THolder<ILogBackendCreator> Slave;
};
