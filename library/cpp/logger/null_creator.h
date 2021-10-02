#pragma once

#include "backend_creator.h"

class TNullLogBackendCreator : public ILogBackendCreator {
public:
    static TFactory::TRegistrator<TNullLogBackendCreator> RegistrarNull;
    static TFactory::TRegistrator<TNullLogBackendCreator> RegistrarDevNull;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
};
