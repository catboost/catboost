#pragma once

#include "file_creator.h"

class TRotatingFileLogBackendCreator : public TFileLogBackendCreator {
public:
    virtual bool Init(const IInitContext& ctx) override;
    static TFactory::TRegistrator<TRotatingFileLogBackendCreator> Registrar;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
    ui64 MaxSizeBytes = Max<ui64>();
    ui64 RotatedFilesCount = Max<ui64>();
};
