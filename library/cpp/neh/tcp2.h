#pragma once

#include <util/datetime/base.h>
#include <util/system/defaults.h>

namespace NNeh {
    //global options
    struct TTcp2Options {
        //connect timeout
        static TDuration ConnectTimeout;

        //input buffer size
        static size_t InputBufferSize;

        //asio client threads
        static size_t AsioClientThreads;

        //asio server threads, - if == 0, use acceptor thread for read/parse incoming requests
        //esle use one thread for accepting + AsioServerThreads for process established tcp connections
        static size_t AsioServerThreads;

        //listen socket queue limit
        static int Backlog;

        //try call non block write to socket from client thread (for decrease latency)
        static bool ClientUseDirectWrite;

        //try call non block write to socket from client thread (for decrease latency)
        static bool ServerUseDirectWrite;

        //expecting receiving request data right after connect or inside receiving request data
        static TDuration ServerInputDeadline;

        //timelimit for sending response data
        static TDuration ServerOutputDeadline;

        //set option, - return false, if option name not recognized
        static bool Set(TStringBuf name, TStringBuf value);
    };

    class IProtocol;

    IProtocol* Tcp2Protocol();
}
