#pragma once

#include "backend_creator.h"

class TNullLogBackendCreator : public TLogBackendCreatorBase {
public:
    TNullLogBackendCreator();
    static TFactory::TRegistrator<TNullLogBackendCreator> RegistrarNull;
    static TFactory::TRegistrator<TNullLogBackendCreator> RegistrarDevNull;
protected:
    virtual void DoToJson(NJson::TJsonValue& value) const override;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
};
