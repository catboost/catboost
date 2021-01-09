#include "network.h"

#include <util/generic/singleton.h>
#include <util/folder/dirut.h>
#include <util/folder/path.h>
#include <util/network/address.h>
#include <util/network/sock.h>
#include <util/system/env.h>
#include <util/system/error.h>
#include <util/system/file_lock.h>
#include <util/system/fs.h>

namespace {
#define Y_VERIFY_SYSERROR(expr)                                           \
    do {                                                                  \
        if (!(expr)) {                                                    \
            Y_FAIL(#expr ", errno=%d", LastSystemError());                \
        }                                                                 \
    } while (false)

    class TPortGuard : public NTesting::IPort {
    public:
        TPortGuard(ui16 port, THolder<TFileLock> lock)
            : Lock_(std::move(lock))
            , Port_(port)
        {
        }

        ~TPortGuard() {
            Y_VERIFY_SYSERROR(NFs::Remove(Lock_->GetName()));
        }

        ui16 Get() override {
            return Port_;
        }

    private:
        THolder<TFileLock> Lock_;
        ui16 Port_;
    };

    class TPortManager {
        static constexpr size_t Retries = 20;
    public:
        TPortManager()
            : SyncDir_(GetEnv("PORT_SYNC_PATH"))
        {
            if (SyncDir_.empty()) {
                SyncDir_ = TFsPath(GetSystemTempDir()) / "yandex_port_locks";
            }
            Y_VERIFY(!SyncDir_.empty());
            NFs::MakeDirectoryRecursive(SyncDir_);
        }

        NTesting::TPortHolder GetFreePort() const {
            for (size_t i = 0; i < Retries; ++i) {
                TInetStreamSocket sock;
                Y_VERIFY_SYSERROR(INVALID_SOCKET != sock);
                Y_VERIFY_SYSERROR(0 == SetSockOpt(sock, SOL_SOCKET, SO_REUSEADDR, 1));

                TSockAddrInet addr{TIpHost{INADDR_ANY}, 0};
                Y_VERIFY_SYSERROR(0 == sock.Bind(&addr));
                auto saddr = NAddr::GetSockAddr(sock);
                Y_VERIFY(AF_INET == saddr->Addr()->sa_family);

                const TIpAddress ipaddr{*reinterpret_cast<const sockaddr_in*>(saddr->Addr())};
                auto port = TryAcquirePort(static_cast<ui16>(ipaddr.Port()));
                if (port) {
                    return NTesting::TPortHolder{std::move(port)};
                }
            }
            Y_FAIL("Cannot get free port!");
        }

        TVector<NTesting::TPortHolder> GetFreePortsRange(size_t count) const {
            Y_VERIFY(count > 0);
            TVector<NTesting::TPortHolder> ports(Reserve(count));
            for (size_t i = 0; i < Retries; ++i) {
                ports.push_back(GetFreePort());

                for (ui16 j = 1; j < count; ++j) {
                    TInetStreamSocket sock(::socket(AF_INET, SOCK_STREAM, 0));
                    Y_VERIFY_SYSERROR(INVALID_SOCKET != static_cast<SOCKET>(sock));
                    Y_VERIFY_SYSERROR(0 == SetSockOpt(sock, SOL_SOCKET, SO_REUSEADDR, 1));

                    ui16 nextPort = static_cast<ui16>(ports.back()) + 1;
                    TSockAddrInet addr{TIpHost{INADDR_ANY}, nextPort};
                    if (0 != sock.Bind(&addr)) {
                        break;
                    }
                    auto port = TryAcquirePort(nextPort);
                    if (!port) {
                        break;
                    }
                    ports.emplace_back(std::move(port));
                }
                if (ports.size() == count) {
                    return ports;
                }
                ports.clear();
            }
            Y_FAIL("Cannot get range of %zu ports!", count);
        }

    private:
        THolder<NTesting::IPort> TryAcquirePort(ui16 port) const {
            auto lock = MakeHolder<TFileLock>(TString(TFsPath(SyncDir_) / ::ToString(port)));
            if (lock->TryAcquire()) {
                return MakeHolder<TPortGuard>(port, std::move(lock));
            }
            return nullptr;
        }

    private:
        TString SyncDir_;
    };
}

namespace NTesting {
    TPortHolder GetFreePort() {
        return Singleton<TPortManager>()->GetFreePort();
    }

    namespace NLegacy {
        TVector<TPortHolder> GetFreePortsRange(size_t count) {
            return Singleton<TPortManager>()->GetFreePortsRange(count);
        }
    }
}
