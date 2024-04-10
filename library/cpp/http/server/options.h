#pragma once

#include <util/network/ip.h>
#include <util/network/init.h>
#include <util/network/address.h>
#include <util/generic/size_literals.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/datetime/base.h>

class IOutputStream;

class THttpServerOptions {
public:
    inline THttpServerOptions(ui16 port = 17000) noexcept
        : Port(port)
    {
    }

    using TBindAddresses = TVector<TNetworkAddress>;
    void BindAddresses(TBindAddresses& ret) const;

    inline THttpServerOptions& AddBindAddress(const TString& address, ui16 port) {
        const TAddr addr = {
            address,
            port,
        };

        BindSockaddr.push_back(addr);
        return *this;
    }

    inline THttpServerOptions& AddBindAddress(const TString& address) {
        return AddBindAddress(address, 0);
    }

    inline THttpServerOptions& EnableKeepAlive(bool enable) noexcept {
        KeepAliveEnabled = enable;

        return *this;
    }

    inline THttpServerOptions& EnableCompression(bool enable) noexcept {
        CompressionEnabled = enable;

        return *this;
    }

    inline THttpServerOptions& EnableRejectExcessConnections(bool enable) noexcept {
        RejectExcessConnections = enable;

        return *this;
    }

    inline THttpServerOptions& EnableReusePort(bool enable) noexcept {
        ReusePort = enable;

        return *this;
    }

    inline THttpServerOptions& EnableReuseAddress(bool enable) noexcept {
        ReuseAddress = enable;

        return *this;
    }

    inline THttpServerOptions& SetThreads(ui32 threads) noexcept {
        nThreads = threads;

        return *this;
    }

    /// Default interface name to bind the server. Used when none of BindAddress are provided.
    inline THttpServerOptions& SetHost(const TString& host) noexcept {
        Host = host;

        return *this;
    }

    /// Default port to bind the server. Used when none of BindAddress are provided.
    inline THttpServerOptions& SetPort(ui16 port) noexcept {
        Port = port;

        return *this;
    }

    inline THttpServerOptions& SetMaxConnections(ui32 mc = 0) noexcept {
        MaxConnections = mc;

        return *this;
    }

    inline THttpServerOptions& SetMaxQueueSize(ui32 mqs = 0) noexcept {
        MaxQueueSize = mqs;

        return *this;
    }

    inline THttpServerOptions& SetClientTimeout(const TDuration& timeout) noexcept {
        ClientTimeout = timeout;

        return *this;
    }

    inline THttpServerOptions& SetListenBacklog(int val) noexcept {
        ListenBacklog = val;

        return *this;
    }

    inline THttpServerOptions& SetOutputBufferSize(size_t val) noexcept {
        OutputBufferSize = val;

        return *this;
    }

    inline THttpServerOptions& SetMaxInputContentLength(ui64 val) noexcept {
        MaxInputContentLength = val;

        return *this;
    }

    inline THttpServerOptions& SetMaxRequestsPerConnection(size_t val) noexcept {
        MaxRequestsPerConnection = val;

        return *this;
    }

    /// Use TElasticQueue instead of TThreadPool for request queues
    inline THttpServerOptions& EnableElasticQueues(bool enable) noexcept {
        UseElasticQueues = enable;

        return *this;
    }

    inline THttpServerOptions& SetThreadsName(const TString& listenThreadName, const TString& requestsThreadName, const TString& failRequestsThreadName) noexcept {
        ListenThreadName = listenThreadName;
        RequestsThreadName = requestsThreadName;
        FailRequestsThreadName = failRequestsThreadName;

        return *this;
    }

    inline THttpServerOptions& SetOneShotPoll(bool v) {
        OneShotPoll = v;

        return *this;
    }

    inline THttpServerOptions& SetListenerThreads(ui32 val) {
        nListenerThreads = val;

        return *this;
    }

    void DebugPrint(IOutputStream& stream) const noexcept;

    struct TAddr {
        TString Addr;
        ui16 Port;
    };

    typedef TVector<TAddr> TAddrs;

    bool KeepAliveEnabled = true;
    bool CompressionEnabled = false;
    bool RejectExcessConnections = false;
    bool ReusePort = false; // set SO_REUSEPORT socket option
    bool ReuseAddress = true; // set SO_REUSEADDR socket option
    TAddrs BindSockaddr;
    ui16 Port = 17000;                  // The port on which to run the web server
    TString Host;                       // DNS entry
    const char* ServerName = "YWS/1.0"; // The Web server name to return in HTTP headers
    ui32 nThreads = 0;                  // Thread count for requests processing
    ui32 MaxQueueSize = 0;              // Max allowed request count in queue
    ui32 nFThreads = 1;
    ui32 MaxFQueueSize = 0;
    ui32 MaxConnections = 100;
    int ListenBacklog = SOMAXCONN;
    ui32 EpollMaxEvents = 1;
    TDuration ClientTimeout;
    size_t OutputBufferSize = 0;
    ui64 MaxInputContentLength = sizeof(size_t) <= 4 ? 2_GB : 64_GB;
    size_t MaxRequestsPerConnection = 0;  // If keep-alive is enabled, request limit before connection is closed
    bool UseElasticQueues = false;

    TDuration PollTimeout; // timeout of TSocketPoller::WaitT call
    TDuration ExpirationTimeout; // drop inactive connections after ExpirationTimeout (should be > 0)

    TString ListenThreadName = "HttpListen";
    TString RequestsThreadName = "HttpServer";
    TString FailRequestsThreadName = "HttpServer";

    bool OneShotPoll = false;
    ui32 nListenerThreads = 1;
};
