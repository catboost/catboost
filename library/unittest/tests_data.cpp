#include <util/folder/dirut.h>
#include <util/generic/vector.h>
#include <util/network/sock.h>
#include <util/system/env.h>
#include "tests_data.h"

#ifdef _win_
const char* DIR_SEPARATORS = "/\\";
#else
const char* DIR_SEPARATORS = "/";
#endif

TString GetArcadiaTestsData() {
    const char* envPath = getenv("ARCADIA_TESTS_DATA_DIR");
    if (envPath != nullptr) {
        return TString(envPath);
    }

    const char* workDir = getcwd(nullptr, 0);
    if (!workDir)
        return "";

    TString path(workDir);
    free((void*)workDir);
    while (!path.empty()) {
        TString dataDir = path + "/arcadia_tests_data";
        if (IsDir(dataDir))
            return dataDir;

        size_t pos = path.find_last_of(DIR_SEPARATORS);
        if (pos == TString::npos)
            pos = 0;
        path.erase(pos);
    }

    return "";
}

TString GetWorkPath() {
    TString envPath = GetEnv("TEST_WORK_PATH");
    if (envPath) {
        return envPath;
    }
    return TString(getcwd(nullptr, 0));
}

class TPortManager::TPortManagerImpl {
public:
    ui16 GetPort(ui16 port) {
        TString env = GetEnv("NO_RANDOM_PORTS");
        if (env && port) {
            return port;
        }

        sockets.push_back(new TInet6StreamSocket());
        TInet6StreamSocket* sock = sockets.back().Get();

        TSockAddrInet6 addr("::", 0);
        SetSockOpt(*sock, SOL_SOCKET, SO_REUSEADDR, 1);

        int ret = sock->Bind(&addr);
        if (ret < 0) {
            ythrow yexception() << "can't bind: " << LastSystemErrorText(-ret);
        }
        return addr.GetPort();
    }

private:
    yvector<THolder<TInet6StreamSocket>> sockets;
};

TPortManager::TPortManager()
    : Impl_(new TPortManagerImpl())
{
}

TPortManager::~TPortManager() {
}

ui16 TPortManager::GetPort(ui16 port) {
    return Impl_->GetPort(port);
}
