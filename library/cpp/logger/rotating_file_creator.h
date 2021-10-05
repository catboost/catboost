#pragma once

#include "file_creator.h"

class TRotatingFileLogBackendCreator : public TFileLogBackendCreator {
public:
    TRotatingFileLogBackendCreator();
    virtual bool Init(const IInitContext& ctx) override;
    static TFactory::TRegistrator<TRotatingFileLogBackendCreator> Registrar;

protected:
    virtual void DoToJson(NJson::TJsonValue& value) const override;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
    ui64 MaxSizeBytes = Max<ui64>();
    ui64 RotatedFilesCount = Max<ui64>();
};
