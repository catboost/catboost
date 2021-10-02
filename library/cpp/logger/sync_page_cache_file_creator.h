#pragma once

#include "file_creator.h"

class TSyncPageCacheFileLogBackendCreator : public TFileLogBackendCreator {
public:
    virtual bool Init(const IInitContext& ctx) override;
    static TFactory::TRegistrator<TSyncPageCacheFileLogBackendCreator> Registrar;

private:
    virtual THolder<TLogBackend> DoCreateLogBackend() const override;
    size_t MaxBufferSize = Max<size_t>();
    size_t MaxPendingCacheSize = Max<size_t>();
};
