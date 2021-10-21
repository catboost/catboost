#pragma once

#include "backend_creator.h"
#include "system.h"

class TSysLogBackendCreator : public TLogBackendCreatorBase {
public:
    TSysLogBackendCreator();
    virtual bool Init(const IInitContext& ctx) override;
    static TFactory::TRegistrator<TSysLogBackendCreator> Registrar;

protected:
    virtual void DoToJson(NJson::TJsonValue& value) const override;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
    TString Ident;
    TSysLogBackend::EFacility Facility = TSysLogBackend::TSYSLOG_LOCAL0;
    int Flags = 0;
};
