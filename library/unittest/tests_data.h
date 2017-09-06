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

    // Gets free TCP port
    ui16 GetPort(ui16 port = 0);

    // Gets free TCP port
    ui16 GetTcpPort(ui16 port = 0);

    // Gets free UDP port
    ui16 GetUdpPort(ui16 port = 0);

    // Gets one free port for use in both TCP and UDP protocols
    ui16 GetTcpAndUdpPort(ui16 port = 0);

private:
    class TPortManagerImpl;
    THolder<TPortManagerImpl> Impl_;
};
