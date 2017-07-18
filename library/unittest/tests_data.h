#pragma once

#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>

TString GetArcadiaTestsData();
TString GetWorkPath();

class TPortManager: public TNonCopyable {
public:
    TPortManager();
    ~TPortManager();
    ui16 GetPort(ui16 port = 0);

private:
    class TPortManagerImpl;
    THolder<TPortManagerImpl> Impl_;
};
