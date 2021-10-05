#include "sync_page_cache_file_creator.h"
#include "sync_page_cache_file.h"

THolder<TLogBackend> TSyncPageCacheFileLogBackendCreator::DoCreateLogBackend() const {
    return MakeHolder<TSyncPageCacheFileLogBackend>(Path, MaxBufferSize, MaxPendingCacheSize);
}


TSyncPageCacheFileLogBackendCreator::TSyncPageCacheFileLogBackendCreator()
    : TFileLogBackendCreator("", "sync_page")
{}

bool TSyncPageCacheFileLogBackendCreator::Init(const IInitContext& ctx) {
    if (!TFileLogBackendCreator::Init(ctx)) {
        return false;
    }
    ctx.GetValue("MaxBufferSize", MaxBufferSize);
    ctx.GetValue("MaxPendingCacheSize", MaxPendingCacheSize);
    return true;
}

ILogBackendCreator::TFactory::TRegistrator<TSyncPageCacheFileLogBackendCreator> TSyncPageCacheFileLogBackendCreator::Registrar("sync_page");

void TSyncPageCacheFileLogBackendCreator::DoToJson(NJson::TJsonValue& value) const {
    TFileLogBackendCreator::DoToJson(value);
    value["MaxBufferSize"] = MaxBufferSize;
    value["MaxPendingCacheSize"] = MaxPendingCacheSize;
}
