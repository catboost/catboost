#pragma once

#include "backend_creator.h"
#include <util/generic/vector.h>

class TCompositeBackendCreator : public ILogBackendCreator {
public:
    virtual bool Init(const IInitContext& ctx) override;
    static TFactory::TRegistrator<TCompositeBackendCreator> Registrar;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
    TVector<THolder<ILogBackendCreator>> Children;
};
