#pragma once

#include "factory.h"
#include "http_common.h"

#include <util/datetime/base.h>
#include <library/cpp/dns/cache.h>
#include <utility>

namespace NNeh {
    IProtocol* Http1Protocol();
    IProtocol* Post1Protocol();
    IProtocol* Full1Protocol();
    IProtocol* Http2Protocol();
    IProtocol* Post2Protocol();
    IProtocol* Full2Protocol();
    IProtocol* UnixSocketGetProtocol();
    IProtocol* UnixSocketPostProtocol();
    IProtocol* UnixSocketFullProtocol();

    //global options
    struct THttp2Options {
        //connect timeout
        static TDuration ConnectTimeout;

        //input and output timeouts
        static TDuration InputDeadline;
        static TDuration OutputDeadline;

        //when detected slow connection, will be runned concurrent parallel connection
        //not used, if SymptomSlowConnect > ConnectTimeout
        static TDuration SymptomSlowConnect;

        //input buffer size
        static size_t InputBufferSize;

        //http client input buffer politic
        static bool KeepInputBufferForCachedConnections;

        //asio threads
        static size_t AsioThreads;

        //asio server threads, - if == 0, use acceptor thread for read/parse incoming requests
        //esle use one thread for accepting + AsioServerThreads for process established tcp connections
        static size_t AsioServerThreads;

        //use ACK for ensure completely sending request (call ioctl() for checking emptiness output buffer)
        //reliable check, but can spend to much time (40ms or like it) (see Wikipedia: TCP delayed acknowledgment)
        //disabling this option reduce sending validation to established connection and written all request data to socket buffer
        static bool EnsureSendingCompleteByAck;

        //listen socket queue limit
        static int Backlog;

        //expecting receiving request data right after connect or inside receiving request data
        static TDuration ServerInputDeadline;

        //timelimit for sending response data
        static TDuration ServerOutputDeadline;

        //expecting receiving request for keep-alived socket
        //(Max - if not reached SoftLimit, Min, if reached Hard limit)
        static TDuration ServerInputDeadlineKeepAliveMax;
        static TDuration ServerInputDeadlineKeepAliveMin;

        //try write data into socket fd in contex handler->SendReply() call
        //(instead moving write job to asio thread)
        //this reduce sys_cpu load (less sys calls), but increase user_cpu and response time
        static bool ServerUseDirectWrite;

        //use http response body as error message
        static bool UseResponseAsErrorMessage;

        //pass all http response headers as error message
        static bool FullHeadersAsErrorMessage;

        //use details (SendError argument) as response body
        static bool ErrorDetailsAsResponseBody;

        //consider responses with 3xx code as successful
        static bool RedirectionNotError;

        //consider response with any code as successful
        static bool AnyResponseIsNotError;

        //enable tcp keepalive for outgoing requests sockets
        static bool TcpKeepAlive;

        //enable limit requests per keep alive connection
        static i32 LimitRequestsPerConnection;

        //enable TCP_QUICKACK
        static bool QuickAck;

        // enable write to socket via ScheduleOp
        static bool UseAsyncSendRequest;

        //  Respect host name/address in THttpServer initialization (pass it it getaddrinfo)
        static bool RespectHostInHttpServerNetworkAddress;

        //set option, - return false, if option name not recognized
        static bool Set(TStringBuf name, TStringBuf value);
    };

    /// if exceed soft limit, reduce quantity unused connections in cache
    void SetHttp2OutputConnectionsLimits(size_t softLimit, size_t hardLimit);

    /// if exceed soft limit, reduce quantity unused connections in cache
    void SetHttp2InputConnectionsLimits(size_t softLimit, size_t hardLimit);

    /// for debug and monitoring purposes
    TAtomicBase GetHttpOutputConnectionCount();
    TAtomicBase GetHttpInputConnectionCount();
    std::pair<size_t, size_t> GetHttpOutputConnectionLimits();

    /// unused input sockets keepalive timeouts
    /// real(used) timeout:
    ///   - max, if not reached soft limit
    ///   - min, if reached hard limit
    ///   - approx. linear changed[max..min], while conn. count in range [soft..hard]
    void SetHttp2InputConnectionsTimeouts(unsigned minSeconds, unsigned maxSeconds);
}
