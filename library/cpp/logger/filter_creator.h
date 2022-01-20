#pragma once

#include "backend_creator.h"
#include "priority.h"

class TFilteredBackendCreator : public ILogBackendCreator {
public:
    TFilteredBackendCreator(THolder<ILogBackendCreator> slave, ELogPriority priority);
    virtual bool Init(const IInitContext& ctx) override;
    virtual void ToJson(NJson::TJsonValue& value) const override;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
    THolder<ILogBackendCreator> Slave;
    ELogPriority Priority;
};
