#include "file_creator.h"
#include "file.h"

TFileLogBackendCreator::TFileLogBackendCreator(const TString& path /*= TString()*/)
    : Path(path)
{}

THolder<TLogBackend> TFileLogBackendCreator::DoCreateLogBackend() const {
    return MakeHolder<TFileLogBackend>(Path);
}

bool TFileLogBackendCreator::Init(const IInitContext& ctx) {
    ctx.GetValue("Path", Path);
    return !!Path;
}

ILogBackendCreator::TFactory::TRegistrator<TFileLogBackendCreator> TFileLogBackendCreator::Registrar("file");
