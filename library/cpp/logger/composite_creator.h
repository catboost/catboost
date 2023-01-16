#pragma once

#include "backend_creator.h"
#include <util/generic/vector.h>

class TCompositeBackendCreator : public TLogBackendCreatorBase {
public:
    TCompositeBackendCreator();
    virtual bool Init(const IInitContext& ctx) override;
    static TFactory::TRegistrator<TCompositeBackendCreator> Registrar;

protected:
    virtual void DoToJson(NJson::TJsonValue& value) const override;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
    TVector<THolder<ILogBackendCreator>> Children;
};
