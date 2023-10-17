#include "network.h"

#include <util/folder/dirut.h>
#include <util/folder/path.h>
#include <util/generic/singleton.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/network/address.h>
#include <util/network/sock.h>
#include <util/random/random.h>
#include <util/stream/file.h>
#include <util/string/split.h>
#include <util/system/env.h>
#include <util/system/error.h>
#include <util/system/file_lock.h>
#include <util/system/fs.h>

#ifdef _darwin_
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

namespace {
#define Y_VERIFY_SYSERROR(expr)                                           \
    do {                                                                  \
        if (!(expr)) {                                                    \
            Y_ABORT(#expr ", errno=%d", LastSystemError());                \
        }                                                                 \
    } while (false)

    class TPortGuard : public NTesting::IPort {
    public:
        TPortGuard(ui16 port, THolder<TFileLock> lock)
            : Lock_(std::move(lock))
            , Port_(port)
        {
        }

        ~TPortGuard() override {
            Y_VERIFY_SYSERROR(NFs::Remove(Lock_->GetName()));
        }

        ui16 Get() override {
            return Port_;
        }

    private:
        THolder<TFileLock> Lock_;
        ui16 Port_;
    };

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

    TVector<std::pair<ui16, ui16>> GetPortRanges() {
        TString givenRange = GetEnv("VALID_PORT_RANGE");
        TVector<std::pair<ui16, ui16>> ranges;
        if (givenRange.Contains(':')) {
            auto res = StringSplitter(givenRange).Split(':').Limit(2).ToList<TString>();
            ranges.emplace_back(FromString<ui16>(res.front()), FromString<ui16>(res.back()));
        } else {
            const ui16 firstValid = 1025;
            const ui16 lastValid = Max<ui16>();

            auto [firstEphemeral, lastEphemeral] = GetEphemeralRange();
            const ui16 firstInvalid = Max(firstEphemeral, firstValid);
            const ui16 lastInvalid = Min(lastEphemeral, lastValid);

            if (firstInvalid > firstValid)
                ranges.emplace_back(firstValid, firstInvalid - 1);
            if (lastInvalid < lastValid)
                ranges.emplace_back(lastInvalid + 1, lastValid);
        }
        return ranges;
    }

    class TPortManager {
        static constexpr size_t Retries = 20;
    public:
        TPortManager()
        {
            InitFromEnv();
        }

        void InitFromEnv() {
            SyncDir_ = TFsPath(GetEnv("PORT_SYNC_PATH"));
            if (!SyncDir_.IsDefined()) {
                SyncDir_ = TFsPath(GetSystemTempDir()) / "testing_port_locks";
            }
            Y_ABORT_UNLESS(SyncDir_.IsDefined());
            NFs::MakeDirectoryRecursive(SyncDir_);

            Ranges_ = GetPortRanges();
            TotalCount_ = 0;
            for (auto [left, right] : Ranges_) {
                TotalCount_ += right - left;
            }
            Y_ABORT_UNLESS(0 != TotalCount_);

            DisableRandomPorts_ = !GetEnv("NO_RANDOM_PORTS").empty();
        }

        NTesting::TPortHolder GetFreePort() const {
            ui16 salt = RandomNumber<ui16>();
            for (ui16 attempt = 0; attempt < TotalCount_; ++attempt) {
                ui16 probe = (salt + attempt) % TotalCount_;
                for (auto [left, right] : Ranges_) {
                    if (probe >= right - left)
                        probe -= right - left;
                    else {
                        probe += left;
                        break;
                    }
                }

                auto port = TryAcquirePort(probe);
                if (port) {
                    return NTesting::TPortHolder{std::move(port)};
                }
            }

            Y_ABORT("Cannot get free port!");
        }

        TVector<NTesting::TPortHolder> GetFreePortsRange(size_t count) const {
            Y_ABORT_UNLESS(count > 0);
            TVector<NTesting::TPortHolder> ports(Reserve(count));
            for (size_t i = 0; i < Retries; ++i) {
                for (auto[left, right] : Ranges_) {
                    if (right - left < count) {
                        continue;
                    }
                    ui16 start = left + RandomNumber<ui16>((right - left) / 2);
                    if (right - start < count) {
                        continue;
                    }
                    for (ui16 probe = start; probe < right; ++probe) {
                        auto port = TryAcquirePort(probe);
                        if (port) {
                            ports.emplace_back(std::move(port));
                        } else {
                            ports.clear();
                        }
                        if (ports.size() == count) {
                            return ports;
                        }
                    }
                    // Can't find required number of ports without gap in the current range
                    ports.clear();
                }
            }
            Y_ABORT("Cannot get range of %zu ports!", count);
        }

        NTesting::TPortHolder GetPort(ui16 port) const {
            if (port && DisableRandomPorts_) {
                auto ackport = TryAcquirePort(port);
                if (ackport) {
                    return NTesting::TPortHolder{std::move(ackport)};
                }
                Y_ABORT("Cannot acquire port %hu!", port);
            }
            return GetFreePort();
        }

    private:
        THolder<NTesting::IPort> TryAcquirePort(ui16 port) const {
            auto lock = MakeHolder<TFileLock>(TString(SyncDir_ / ::ToString(port)));
            if (!lock->TryAcquire()) {
                return nullptr;
            }

            TInet6StreamSocket sock;
            Y_VERIFY_SYSERROR(INVALID_SOCKET != static_cast<SOCKET>(sock));

            TSockAddrInet6 addr("::", port);
            if (sock.Bind(&addr) != 0) {
                lock->Release();
                Y_ABORT_UNLESS(EADDRINUSE == LastSystemError(), "unexpected error: %d, port: %d", LastSystemError(), port);
                return nullptr;
            }
            return MakeHolder<TPortGuard>(port, std::move(lock));
        }

    private:
        TFsPath SyncDir_;
        TVector<std::pair<ui16, ui16>> Ranges_;
        size_t TotalCount_;
        bool DisableRandomPorts_;
    };
}

namespace NTesting {
    void InitPortManagerFromEnv() {
        Singleton<TPortManager>()->InitFromEnv();
    }

    TPortHolder GetFreePort() {
        return Singleton<TPortManager>()->GetFreePort();
    }

    namespace NLegacy {
        TPortHolder GetPort( ui16 port ) {
            return Singleton<TPortManager>()->GetPort(port);
        }
        TVector<TPortHolder> GetFreePortsRange(size_t count) {
            return Singleton<TPortManager>()->GetFreePortsRange(count);
        }
    }

    IOutputStream& operator<<(IOutputStream& out, const TPortHolder& port) {
        return out << static_cast<ui16>(port);
    }
}
