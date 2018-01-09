#include <util/folder/dirut.h>
#include <util/generic/vector.h>
#include <util/network/sock.h>
#include <util/system/env.h>
#include <util/system/file_lock.h>
#include <util/system/fs.h>
#include <util/system/mutex.h>

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

TFsPath GetYaPath() {
    TString envPath = GetEnv("YA_CACHE_DIR");
    if (!envPath) {
        envPath = GetHomeDir() + "/.ya";
    }
    return envPath;
}

TFsPath GetOutputPath() {
    return GetWorkPath() + "/testing_out_stuff";
}

class TPortManager::TPortManagerImpl {
    class TPortGuard {
    public:
        using TPtr = TSimpleSharedPtr<TPortGuard>;

        TPortGuard(const TFsPath& root, const ui16 port);
        ~TPortGuard();

        bool IsLocked() const;
        ui16 GetPort() const;
    private:
        TFsPath Path;
        ui16 Port;
        TFileLock Lock;
        TSimpleSharedPtr<TInet6StreamSocket> Socket;
        bool Locked = false;
    };

public:
    TPortManagerImpl(const TString& syncDir) {
        SyncDir = GetEnv("PORT_SYNC_PATH", syncDir);
        if (IsSyncDirSet())
            NFs::MakeDirectoryRecursive(SyncDir);
    }

    ui16 GetUdpPort(ui16 port) {
        return GetPort<TInet6DgramSocket>(port);
    }

    ui16 GetTcpPort(ui16 port) {
        return GetPort<TInet6StreamSocket>(port);
    }

    template <class TSocketType>
    ui16 GetPort(ui16 port) {
        if (port && NoRandomPorts()) {
            return port;
        }

        size_t retries = 20;
        int bind_error = 0;

        while (retries--) {
            THolder<TSocketType> sock = new TSocketType();
            SetReuseAddressAndPort(*sock);

            TSockAddrInet6 addr("::", 0);
            bind_error = sock->Bind(&addr);
            if (bind_error < 0)
                continue;

            ui16 port = addr.GetPort();
            if (IsSyncDirSet() && !LockPort(port))
                continue;

            Sockets.push_back(std::move(sock));
            return port;
        }
        if (bind_error)
            ythrow yexception() << "Can't bind: " << LastSystemErrorText(-bind_error);
        ythrow yexception() << "Failed to find port";
    }

    ui16 GetTcpAndUdpPort(ui16 port) {
        if (port && NoRandomPorts()) {
            return port;
        }

        size_t retries = 20;
        while (retries--) {
            // 1. Get random free TCP port. Ports are guaranteed to be different with
            //    ports given by get_tcp_port() and other get_tcp_and_udp_port() methods.
            // 2. Bind the same UDP port without SO_REUSEADDR to avoid race with get_udp_port() method:
            //    if get_udp_port() from other thread/process gets this port, bind() fails; if bind()
            //    succeeds, then get_udp_port() from other thread/process gives other port.
            // 3. Set SO_REUSEADDR option to let use this UDP port from test.
            const ui16 resultPort = GetTcpPort(0);
            THolder<TInet6DgramSocket> sock = new TInet6DgramSocket();
            TSockAddrInet6 addr("::", resultPort);
            const int ret = sock->Bind(&addr);
            if (ret < 0) {
                Sockets.pop_back();
                continue;
            }
            SetReuseAddressAndPort(*sock);
            Sockets.push_back(std::move(sock));
            return resultPort;
        }
        ythrow yexception() << "Failed to find port";
    }

    ui16 GetPortsRange(const ui16 startPort, const ui16 range) {
        Y_ENSURE(range > 0);
        TGuard<TMutex> g(Lock);

        TVector<TPortGuard::TPtr> candidates;

        for (ui16 port = startPort; candidates.size() < range && port < Max<ui16>() - range; ++port) {
            TPortGuard::TPtr guard(new TPortGuard(SyncDir, port));
            if (!guard->IsLocked()) {
                candidates.clear();
            } else {
                candidates.push_back(guard);
            }
        }

        Y_ENSURE(candidates.size() == range);
        ReservedPorts.insert(ReservedPorts.end(), candidates.begin(), candidates.end());
        return candidates.front()->GetPort();
    }

private:
    static bool NoRandomPorts() {
        return !GetEnv("NO_RANDOM_PORTS").empty();
    }

    bool IsSyncDirSet() {
        return !SyncDir.empty();
    }

    bool LockPort(ui16 port) {
        TGuard<TMutex> g(Lock);
        TPortGuard::TPtr guard(new TPortGuard(SyncDir, port));
        bool locked = guard->IsLocked();
        if (locked)
            ReservedPorts.push_back(guard);
        return locked;
    }

private:
    TVector<THolder<TBaseSocket>> Sockets;
    TString SyncDir;
    TVector<TPortGuard::TPtr> ReservedPorts;
    TMutex Lock;
};

TPortManager::TPortManager(const TString& syncDir)
    : Impl_(new TPortManagerImpl(syncDir))
{
}

TPortManager::~TPortManager() {
}

ui16 TPortManager::GetPort(ui16 port) {
    return Impl_->GetTcpPort(port);
}

ui16 TPortManager::GetTcpPort(ui16 port) {
    return Impl_->GetTcpPort(port);
}

ui16 TPortManager::GetUdpPort(ui16 port) {
    return Impl_->GetUdpPort(port);
}

ui16 TPortManager::GetTcpAndUdpPort(ui16 port) {
    return Impl_->GetTcpAndUdpPort(port);
}

ui16 TPortManager::GetPortsRange(const ui16 startPort, const ui16 range) {
    return Impl_->GetPortsRange(startPort, range);
}

TPortManager::TPortManagerImpl::TPortGuard::TPortGuard(const TFsPath& root, const ui16 port)
    : Path(root / ::ToString(port))
    , Port(port)
    , Lock(Path)
{
    if (Lock.TryAcquire()) {
        Socket.Reset(new TInet6StreamSocket());

        TSockAddrInet6 addr("::", port);
        SetReuseAddressAndPort(*Socket);

        if (Socket->Bind(&addr) == 0) {
            Locked = true;
        }
    }
}

bool TPortManager::TPortManagerImpl::TPortGuard::IsLocked() const {
    return Locked;
}

ui16 TPortManager::TPortManagerImpl::TPortGuard::GetPort() const {
    return Port;
}

TPortManager::TPortManagerImpl::TPortGuard::~TPortGuard() {
    if (Locked) {
        NFs::Remove(Path.GetPath());
        Lock.Release();
    }
}
