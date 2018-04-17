#include <util/folder/dirut.h>
#include <util/generic/vector.h>
#include <util/network/sock.h>
#include <util/random/random.h>
#include <util/stream/file.h>
#include <util/system/env.h>
#include <util/system/file_lock.h>
#include <util/system/fs.h>
#include <util/system/mutex.h>
#include <util/system/env.h>

#ifdef _darwin_
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include "tests_data.h"

#ifdef _win_
const char* DIR_SEPARATORS = "/\\";
#else
const char* DIR_SEPARATORS = "/";
#endif

TString GetArcadiaTestsData() {
    TString envPath = GetEnv("ARCADIA_TESTS_DATA_DIR");
    if (envPath) {
        return envPath;
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
    char* cwd = getcwd(nullptr, 0);
    TString workPath = TString(cwd);
    free(cwd);
    return workPath;
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
    TPortManagerImpl(const TString& syncDir)
        : ValidPortsCount(0)
    {
        SyncDir = GetEnv("PORT_SYNC_PATH", syncDir);
        if (IsSyncDirSet())
            NFs::MakeDirectoryRecursive(SyncDir);
        InitValidPortRange();
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

        ui16 salt = RandomNumber<ui16>();
        for (ui16 attempt = 0; attempt < ValidPortsCount; ++attempt) {
            port = (salt + attempt) % ValidPortsCount;

            for (auto&& range : ValidPortRanges) {
                if (port >= range.second - range.first + 1)
                    port -= range.second - range.first + 1;
                else {
                    port += range.first;
                    break;
                }
            }

            THolder<TSocketType> sock = new TSocketType();

            TSockAddrInet6 addr("::", port);
            if (sock->Bind(&addr) < 0)
                continue;
            SetReuseAddressAndPort(*sock);

            if (IsSyncDirSet() && !LockPort(port))
                continue;

            {
                TGuard<TMutex> g(Lock);
                Sockets.push_back(std::move(sock));
            }
            return port;
        }
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

    void InitValidPortRange() {
        const ui16 first_valid = 1025;
        const ui16 last_valid = (1 << 16) - 1;

        auto ephemeral = GetEphemeralRange();
        const ui16 first_invalid = std::max(ephemeral.first, first_valid);
        const ui16 last_invalid = std::min(ephemeral.second, last_valid);

        ValidPortRanges.clear();
        if (first_invalid > first_valid)
            ValidPortRanges.emplace_back(first_valid, first_invalid - 1);
        if (last_invalid < last_valid)
            ValidPortRanges.emplace_back(last_invalid + 1, last_valid);

        ValidPortsCount = 0;
        for (auto&& range : ValidPortRanges)
            ValidPortsCount += range.second - range.first + 1;

        Y_VERIFY(ValidPortsCount);
    }

    std::pair<ui16, ui16> GetEphemeralRange() {
        // IANA suggestion
        std::pair<ui16, ui16> pair{(1 << 15) + (1 << 14), (1 << 16) - 1};
#ifdef _linux_
        if (NFs::Exists("/proc/sys/net/ipv4/ip_local_port_range")) {
            TIFStream fileStream("/proc/sys/net/ipv4/ip_local_port_range");
            fileStream >> pair.first >> pair.second;
        }
#endif
#ifdef _darwin_
        ui32 first, last;
        size_t size;
        sysctlbyname("net.inet.ip.portrange.first", &first, &size, NULL, 0);
        sysctlbyname("net.inet.ip.portrange.last", &last, &size, NULL, 0);
        pair.first = first;
        pair.second = last;
#endif
        return pair;
    }

private:
    TVector<THolder<TBaseSocket>> Sockets;
    TString SyncDir;
    TVector<TPortGuard::TPtr> ReservedPorts;
    TMutex Lock;
    ui16 ValidPortsCount;
    TVector<std::pair<ui16, ui16>> ValidPortRanges;
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
