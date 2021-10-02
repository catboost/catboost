#pragma once

#include "backend_creator.h"

class TFileLogBackendCreator : public ILogBackendCreator {
public:
    TFileLogBackendCreator(const TString& path = TString());
    virtual bool Init(const IInitContext& ctx) override;
    static TFactory::TRegistrator<TFileLogBackendCreator> Registrar;

protected:
    TString Path;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
};
