#pragma once

#include <library/cpp/netliba/socket/socket.h>

namespace NNetliba_v12 {
    using NNetlibaSocket::AddSockAuxData;
    using NNetlibaSocket::CTRL_BUFFER_SIZE;
    using NNetlibaSocket::CreateIoVec;
    using NNetlibaSocket::CreateRecvMsgHdr;
    using NNetlibaSocket::CreateSendMsgHdr;
    using NNetlibaSocket::CreateTos;
    using NNetlibaSocket::TIoVec;
    using NNetlibaSocket::TMMsgHdr;
    using NNetlibaSocket::TMsgHdr;

    using NNetlibaSocket::CreateDualStackSocket;
    using NNetlibaSocket::CreateSocket;
    using NNetlibaSocket::ISocket;

    using NNetlibaSocket::EFragFlag;
    using NNetlibaSocket::FF_ALLOW_FRAG;
    using NNetlibaSocket::FF_DONT_FRAG;
}
