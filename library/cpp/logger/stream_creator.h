#pragma once

#include "backend_creator.h"

class TCerrLogBackendCreator : public ILogBackendCreator {
public:
    static TFactory::TRegistrator<TCerrLogBackendCreator> RegistrarCerr;
    static TFactory::TRegistrator<TCerrLogBackendCreator> RegistrarConsole;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
};

class TCoutLogBackendCreator : public ILogBackendCreator {
public:
    static TFactory::TRegistrator<TCoutLogBackendCreator> Registrar;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
};
