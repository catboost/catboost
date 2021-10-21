#pragma once

#include "backend_creator.h"

class TFileLogBackendCreator : public TLogBackendCreatorBase {
public:
    TFileLogBackendCreator(const TString& path = TString(), const TString& type = "file");
    virtual bool Init(const IInitContext& ctx) override;
    static TFactory::TRegistrator<TFileLogBackendCreator> Registrar;

protected:
    virtual void DoToJson(NJson::TJsonValue& value) const override;
    TString Path;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
};
