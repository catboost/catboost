#include "local_host.h"
#include "exit.h"
#include "fork_aware_spin_lock.h"

#include <library/cpp/yt/misc/immortal.h>

#include <util/generic/hash_set.h>

#include <atomic>
#include <cstring>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace {

class TStaticName
{
public:
    TStringBuf Read() const noexcept
    {
        // Pairs with the release store in Write.
        char* ptr = Ptr_.load(std::memory_order::acquire);
        return ptr ? ptr : Buffer_;
    }

    void Write(TStringBuf value) noexcept
    {
        char* ptr = Ptr_.load(std::memory_order::relaxed);
        ptr = ptr ? ptr : Buffer_;

        if (TStringBuf(ptr) == value) {
            // No changes; just return.
            return;
        }

        ptr = ptr + strlen(ptr) + 1;

        if (ptr + value.length() + 1 >= Buffer_ + BufferSize) {
            AbortProcessDramatically(
                EProcessExitCode::InternalError,
                "TStaticName is out of buffer space");
        }

        ::memcpy(ptr, value.data(), value.length());
        *(ptr + value.length()) = 0;

        Ptr_.store(ptr, std::memory_order::release);
    }

private:
    static constexpr size_t BufferSize = 1024;
    char Buffer_[BufferSize] = "(unknown)";
    std::atomic<char*> Ptr_;
};

// All static variables below must be constinit.
constinit TStaticName LocalHostName;
constinit TStaticName LocalYPCluster;

constexpr size_t MaxYPClusterNameSize = 32;

} // namespace

////////////////////////////////////////////////////////////////////////////////

TStringBuf GetLocalHostNameRaw() noexcept
{
    return LocalHostName.Read();
}

TStringBuf GetLocalYPClusterRaw() noexcept
{
    return LocalYPCluster.Read();
}

void SetLocalHostName(TStringBuf hostName) noexcept
{
    static YT_DECLARE_SPIN_LOCK(TForkAwareSpinLock, Lock);
    auto guard = Guard(Lock);

    LocalHostName.Write(hostName);

    if (auto ypCluster = InferYPClusterFromHostNameRaw(hostName)) {
        LocalYPCluster.Write(*ypCluster);
    }
}

std::string GetLocalHostName()
{
    return std::string(GetLocalHostNameRaw());
}

std::string GetLocalYPCluster()
{
    return std::string(GetLocalYPClusterRaw());
}

std::optional<TStringBuf> InferYPClusterFromHostNameRaw(TStringBuf hostName)
{
    auto start = hostName.find_first_of('.');
    if (start == TStringBuf::npos) {
        return {};
    }
    auto end = hostName.find_first_of('.', start + 1);
    if (end == TStringBuf::npos) {
        return {};
    }
    auto cluster = hostName.substr(start + 1, end - start - 1);
    if (cluster.empty()) {
        return {};
    }
    if (cluster.length() > MaxYPClusterNameSize) {
        return {};
    }
    return {cluster};
}

TStringBuf InternHostName(TStringBuf hostName)
{
    static YT_DECLARE_SPIN_LOCK(TForkAwareSpinLock, Lock);
    static TImmortal<THashSet<TStringBuf>> HostNames;

    auto guard = Guard(Lock);

    if (auto it = HostNames->find(hostName); it != HostNames->end()) {
        return *it;
    }

    auto* data = new char[hostName.size() + 1];
    std::memcpy(data, hostName.data(), hostName.size());
    data[hostName.size()] = '\0';

    TStringBuf result(data, hostName.size());
    HostNames->insert(result);
    return result;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
