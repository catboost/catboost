#pragma once
#include <library/netliba/socket/socket.h>

namespace NNetliba_v12
{
using NNetlibaSocket::CTRL_BUFFER_SIZE;
using NNetlibaSocket::TIoVec;
using NNetlibaSocket::TMsgHdr;
using NNetlibaSocket::TMMsgHdr;
using NNetlibaSocket::CreateTos;
using NNetlibaSocket::CreateIoVec;
using NNetlibaSocket::CreateSendMsgHdr;
using NNetlibaSocket::CreateRecvMsgHdr;
using NNetlibaSocket::AddSockAuxData;

using NNetlibaSocket::ISocket;
using NNetlibaSocket::CreateSocket;
using NNetlibaSocket::CreateDualStackSocket;

using NNetlibaSocket::EFragFlag;
using NNetlibaSocket::FF_ALLOW_FRAG;
using NNetlibaSocket::FF_DONT_FRAG;
}
