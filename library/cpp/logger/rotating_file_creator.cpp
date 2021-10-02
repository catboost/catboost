#include "rotating_file_creator.h"
#include "rotating_file.h"

THolder<TLogBackend> TRotatingFileLogBackendCreator::DoCreateLogBackend() const {
    return MakeHolder<TRotatingFileLogBackend>(Path, MaxSizeBytes, RotatedFilesCount);
}

bool TRotatingFileLogBackendCreator::Init(const IInitContext& ctx) {
    if (!TFileLogBackendCreator::Init(ctx)) {
        return false;
    }
    ctx.GetValue("MaxSizeBytes", MaxSizeBytes);
    ctx.GetValue("RotatedFilesCount", RotatedFilesCount);
    return true;
}

ILogBackendCreator::TFactory::TRegistrator<TRotatingFileLogBackendCreator> TRotatingFileLogBackendCreator::Registrar("rotating");
