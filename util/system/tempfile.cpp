#include "tempfile.h"

TTempFileHandle::TTempFileHandle()
    : TTempFile(MakeTempName())
    , TFile(CreateFile())
{
}

TTempFileHandle::TTempFileHandle(const TString& fname)
    : TTempFile(fname)
    , TFile(CreateFile())
{
}

TTempFileHandle TTempFileHandle::InCurrentDir(const TString& filePrefix, const TString& extension) {
    return TTempFileHandle(MakeTempName(".", filePrefix.c_str(), extension.c_str()));
}

TTempFileHandle TTempFileHandle::InDir(const TFsPath& dirPath, const TString& filePrefix, const TString& extension) {
    return TTempFileHandle(MakeTempName(dirPath.c_str(), filePrefix.c_str(), extension.c_str()));
}

TFile TTempFileHandle::CreateFile() const {
    return TFile(Name(), CreateAlways | RdWr);
}
