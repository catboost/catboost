#include "mktemp.h"
#include "tempfile.h"

TFile TTempFileHandle::CreateFile() const {
    return TFile(Name(), CreateAlways | RdWr);
}

TTempFileHandle::TTempFileHandle(const TString& fname)
    : TTempFile(fname)
    , TFile(CreateFile())
{
}

TTempFileHandle::TTempFileHandle()
    : TTempFile(MakeTempName())
    , TFile(CreateFile())
{
}
