#include "tempfile.h"

#include "error.h"

#include <util/generic/yexception.h>

TTempFile& TTempFile::operator=(TTempFile&& rhs) {
    if (Name_.Defined()) {
        Y_ENSURE(NFs::Remove(*Name_), "Removing \"" << *Name_ << "\" failed: " << LastSystemErrorText());
    }
    Name_ = std::move(rhs.Name_);
    rhs.Name_.Clear();
    return *this;
}

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
