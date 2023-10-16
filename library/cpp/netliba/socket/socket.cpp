#include "stdafx.h"
#include <util/datetime/cputimer.h>
#include <util/draft/holder_vector.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/network/init.h>
#include <util/network/poller.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/byteorder.h>
#include <util/system/defaults.h>
#include <util/system/error.h>
#include <util/system/event.h>
#include <util/system/thread.h>
#include <util/system/yassert.h>
#include <util/system/rwlock.h>
#include <util/system/env.h>

#include "socket.h"
#include "packet_queue.h"
#include "udp_recv_packet.h"

#include <array>
#include <stdlib.h>

///////////////////////////////////////////////////////////////////////////////

#ifndef _win_
#include <netinet/in.h>
#endif

#ifdef _linux_
#include <dlfcn.h> // dlsym
#endif

template <class T>
static T GetAddressOf(const char* name) {
#ifdef _linux_
    if (!GetEnv("DISABLE_MMSG")) {
        return (T)dlsym(RTLD_DEFAULT, name);
    }
#endif
    Y_UNUSED(name);
    return nullptr;
}

///////////////////////////////////////////////////////////////////////////////

namespace NNetlibaSocket {
    ///////////////////////////////////////////////////////////////////////////////

    struct timespec; // we use it only as NULL pointer
    typedef int (*TSendMMsgFunc)(SOCKET, TMMsgHdr*, unsigned int, unsigned int);
    typedef int (*TRecvMMsgFunc)(SOCKET, TMMsgHdr*, unsigned int, unsigned int, timespec*);

    static const TSendMMsgFunc SendMMsgFunc = GetAddressOf<TSendMMsgFunc>("sendmmsg");
    static const TRecvMMsgFunc RecvMMsgFunc = GetAddressOf<TRecvMMsgFunc>("recvmmsg");

    ///////////////////////////////////////////////////////////////////////////////

    bool ReadTos(const TMsgHdr& msgHdr, ui8* tos) {
#ifdef _win_
        Y_UNUSED(msgHdr);
        Y_UNUSED(tos);
        return false;
#else
        cmsghdr* cmsg = CMSG_FIRSTHDR(&msgHdr);
        if (!cmsg)
            return false;
        //Y_ASSERT(cmsg->cmsg_level == IPPROTO_IPV6);
        //Y_ASSERT(cmsg->cmsg_type == IPV6_TCLASS);
        if (cmsg->cmsg_len != CMSG_LEN(sizeof(int)))
            return false;
        *tos = *(ui8*)CMSG_DATA(cmsg);
        return true;
#endif
    }

    bool ExtractDestinationAddress(TMsgHdr& msgHdr, sockaddr_in6* addrBuf) {
        Zero(*addrBuf);
#ifdef _win_
        Y_UNUSED(msgHdr);
        Y_UNUSED(addrBuf);
        return false;
#else
        cmsghdr* cmsg;
        for (cmsg = CMSG_FIRSTHDR(&msgHdr); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msgHdr, cmsg)) {
            if ((cmsg->cmsg_level == IPPROTO_IPV6) && (cmsg->cmsg_type == IPV6_PKTINFO)) {
                in6_pktinfo* i = (in6_pktinfo*)CMSG_DATA(cmsg);
                addrBuf->sin6_addr = i->ipi6_addr;
                addrBuf->sin6_family = AF_INET6;
                return true;
            }
        }
        return false;
#endif
    }

    // all send and recv methods are thread safe!
    class TAbstractSocket: public ISocket {
    private:
        SOCKET S;
        mutable TSocketPoller Poller;
        sockaddr_in6 SelfAddress;

        int SendSysSocketSize;
        int SendSysSocketSizePrev;

        int CreateSocket(int netPort);
        int DetectSelfAddress();

    protected:
        int SetSockOpt(int level, int option_name, const void* option_value, socklen_t option_len);

        int OpenImpl(int port);
        void CloseImpl();

        void WaitImpl(float timeoutSec) const;
        void CancelWaitImpl(const sockaddr_in6* address = nullptr); // NULL means "self"

        ssize_t RecvMsgImpl(TMsgHdr* hdr, int flags);
        TUdpRecvPacket* RecvImpl(TUdpHostRecvBufAlloc* buf, sockaddr_in6* srcAddr, sockaddr_in6* dstAddr);
        int RecvMMsgImpl(TMMsgHdr* msgvec, unsigned int vlen, unsigned int flags, timespec* timeout);

        bool IsFragmentationForbiden();
        void ForbidFragmentation();
        void EnableFragmentation();

        //Shared state for setsockopt. Forbid simultaneous transfer while sender asking for specific options (i.e. DONOT_FRAG)
        TRWMutex Mutex;
        TAtomic RecvLag = 0;

    public:
        TAbstractSocket();
        ~TAbstractSocket() override;
#ifdef _unix_
        void Reset(const TAbstractSocket& rhv);
#endif

        bool IsValid() const override;

        const sockaddr_in6& GetSelfAddress() const override;
        int GetNetworkOrderPort() const override;
        int GetPort() const override;

        int GetSockOpt(int level, int option_name, void* option_value, socklen_t* option_len) override;

        // send all packets to this and only this address by default
        int Connect(const struct sockaddr* address, socklen_t address_len) override;

        void CancelWaitHost(const sockaddr_in6 addr) override;

        bool IsSendMMsgSupported() const override;
        int SendMMsg(TMMsgHdr* msgvec, unsigned int vlen, unsigned int flags) override;
        ssize_t SendMsg(const TMsgHdr* hdr, int flags, const EFragFlag frag) override;
        bool IncreaseSendBuff() override;
        int GetSendSysSocketSize() override;
        void SetRecvLagTime(NHPTimer::STime time) override;
    };

    TAbstractSocket::TAbstractSocket()
        : S(INVALID_SOCKET)
        , SendSysSocketSize(0)
        , SendSysSocketSizePrev(0)
    {
        Zero(SelfAddress);
    }

    TAbstractSocket::~TAbstractSocket() {
        CloseImpl();
    }

#ifdef _unix_
    void TAbstractSocket::Reset(const TAbstractSocket& rhv) {
        Close();
        S = dup(rhv.S);
        SelfAddress = rhv.SelfAddress;
    }
#endif

    int TAbstractSocket::CreateSocket(int netPort) {
        if (IsValid()) {
            Y_ASSERT(0);
            return 0;
        }
        S = socket(AF_INET6, SOCK_DGRAM, IPPROTO_UDP);
        if (S == INVALID_SOCKET) {
            return -1;
        }
        {
            int flag = 0;
            Y_ABORT_UNLESS(SetSockOpt(IPPROTO_IPV6, IPV6_V6ONLY, (const char*)&flag, sizeof(flag)) == 0, "IPV6_V6ONLY failed");
        }
        {
            int flag = 1;
            Y_ABORT_UNLESS(SetSockOpt(SOL_SOCKET, SO_REUSEADDR, (const char*)&flag, sizeof(flag)) == 0, "SO_REUSEADDR failed");
        }
#if defined(_win_)
        unsigned long dummy = 1;
        ioctlsocket(S, FIONBIO, &dummy);
#else
        Y_ABORT_UNLESS(fcntl(S, F_SETFL, O_NONBLOCK) == 0, "fnctl failed: %s (errno = %d)", LastSystemErrorText(), LastSystemError());
        Y_ABORT_UNLESS(fcntl(S, F_SETFD, FD_CLOEXEC) == 0, "fnctl failed: %s (errno = %d)", LastSystemErrorText(), LastSystemError());
        {
            int flag = 1;
#ifndef IPV6_RECVPKTINFO /* Darwin platforms require this */
            Y_ABORT_UNLESS(SetSockOpt(IPPROTO_IPV6, IPV6_PKTINFO, (const char*)&flag, sizeof(flag)) == 0, "IPV6_PKTINFO failed");
#else
            Y_ABORT_UNLESS(SetSockOpt(IPPROTO_IPV6, IPV6_RECVPKTINFO, (const char*)&flag, sizeof(flag)) == 0, "IPV6_RECVPKTINFO failed");
#endif
        }
#endif

        Poller.WaitRead(S, nullptr);

        {
            // bind socket
            sockaddr_in6 name;
            Zero(name);
            name.sin6_family = AF_INET6;
            name.sin6_addr = in6addr_any;
            name.sin6_port = netPort;
            if (bind(S, (sockaddr*)&name, sizeof(name)) != 0) {
                fprintf(stderr, "netliba_socket could not bind to port %d: %s (errno = %d)\n", InetToHost((ui16)netPort), LastSystemErrorText(), LastSystemError());
                CloseImpl(); // we call this CloseImpl after Poller initialization
                return -1;
            }
        }
        //Default behavior is allowing fragmentation (according to netliba v6 behavior)
        //If we want to sent packet with DF flag we have to use SendMsg()
        EnableFragmentation();

        {
            socklen_t sz = sizeof(SendSysSocketSize);
            if (GetSockOpt(SOL_SOCKET, SO_SNDBUF, &SendSysSocketSize, &sz)) {
                fprintf(stderr, "Can`t get SO_SNDBUF");
            }
        }
        return 0;
    }

    bool TAbstractSocket::IsValid() const {
        return S != INVALID_SOCKET;
    }

    int TAbstractSocket::DetectSelfAddress() {
        socklen_t nameLen = sizeof(SelfAddress);
        if (getsockname(S, (sockaddr*)&SelfAddress, &nameLen) != 0) { // actually we use only sin6_port
            return -1;
        }
        Y_ASSERT(SelfAddress.sin6_family == AF_INET6);
        SelfAddress.sin6_addr = in6addr_loopback;
        return 0;
    }

    const sockaddr_in6& TAbstractSocket::GetSelfAddress() const {
        return SelfAddress;
    }

    int TAbstractSocket::GetNetworkOrderPort() const {
        return SelfAddress.sin6_port;
    }

    int TAbstractSocket::GetPort() const {
        return InetToHost((ui16)SelfAddress.sin6_port);
    }

    int TAbstractSocket::SetSockOpt(int level, int option_name, const void* option_value, socklen_t option_len) {
        const int rv = setsockopt(S, level, option_name, (const char*)option_value, option_len);
        Y_DEBUG_ABORT_UNLESS(rv == 0, "SetSockOpt failed: %s (errno = %d)", LastSystemErrorText(), LastSystemError());
        return rv;
    }

    int TAbstractSocket::GetSockOpt(int level, int option_name, void* option_value, socklen_t* option_len) {
        const int rv = getsockopt(S, level, option_name, (char*)option_value, option_len);
        Y_DEBUG_ABORT_UNLESS(rv == 0, "GetSockOpt failed: %s (errno = %d)", LastSystemErrorText(), LastSystemError());
        return rv;
    }

    bool TAbstractSocket::IsFragmentationForbiden() {
#if defined(_win_)
        DWORD flag = 0;
        socklen_t sz = sizeof(flag);
        Y_ABORT_UNLESS(GetSockOpt(IPPROTO_IP, IP_DONTFRAGMENT, (char*)&flag, &sz) == 0, "");
        return flag;
#elif defined(_linux_)
        int flag = 0;
        socklen_t sz = sizeof(flag);
        Y_ABORT_UNLESS(GetSockOpt(IPPROTO_IPV6, IPV6_MTU_DISCOVER, (char*)&flag, &sz) == 0, "");
        return flag == IPV6_PMTUDISC_DO;
#elif !defined(_darwin_)
        int flag = 0;
        socklen_t sz = sizeof(flag);
        Y_ABORT_UNLESS(GetSockOpt(IPPROTO_IPV6, IPV6_DONTFRAG, (char*)&flag, &sz) == 0, "");
        return flag;
#endif
        return false;
    }

    void TAbstractSocket::ForbidFragmentation() {
    // do not fragment ping packets
#if defined(_win_)
        DWORD flag = 1;
        SetSockOpt(IPPROTO_IP, IP_DONTFRAGMENT, (const char*)&flag, sizeof(flag));
#elif defined(_linux_)
        int flag = IP_PMTUDISC_DO;
        SetSockOpt(IPPROTO_IP, IP_MTU_DISCOVER, (const char*)&flag, sizeof(flag));

        flag = IPV6_PMTUDISC_DO;
        SetSockOpt(IPPROTO_IPV6, IPV6_MTU_DISCOVER, (const char*)&flag, sizeof(flag));
#elif !defined(_darwin_)
        int flag = 1;
        //SetSockOpt(IPPROTO_IP, IP_DONTFRAG, (const char*)&flag, sizeof(flag));
        SetSockOpt(IPPROTO_IPV6, IPV6_DONTFRAG, (const char*)&flag, sizeof(flag));
#endif
    }

    void TAbstractSocket::EnableFragmentation() {
#if defined(_win_)
        DWORD flag = 0;
        SetSockOpt(IPPROTO_IP, IP_DONTFRAGMENT, (const char*)&flag, sizeof(flag));
#elif defined(_linux_)
        int flag = IP_PMTUDISC_WANT;
        SetSockOpt(IPPROTO_IP, IP_MTU_DISCOVER, (const char*)&flag, sizeof(flag));

        flag = IPV6_PMTUDISC_WANT;
        SetSockOpt(IPPROTO_IPV6, IPV6_MTU_DISCOVER, (const char*)&flag, sizeof(flag));
#elif !defined(_darwin_)
        int flag = 0;
        //SetSockOpt(IPPROTO_IP, IP_DONTFRAG, (const char*)&flag, sizeof(flag));
        SetSockOpt(IPPROTO_IPV6, IPV6_DONTFRAG, (const char*)&flag, sizeof(flag));
#endif
    }

    int TAbstractSocket::Connect(const sockaddr* address, socklen_t address_len) {
        Y_ASSERT(IsValid());
        return connect(S, address, address_len);
    }

    void TAbstractSocket::CancelWaitHost(const sockaddr_in6 addr) {
        CancelWaitImpl(&addr);
    }

    bool TAbstractSocket::IsSendMMsgSupported() const {
        return SendMMsgFunc != nullptr;
    }

    int TAbstractSocket::SendMMsg(TMMsgHdr* msgvec, unsigned int vlen, unsigned int flags) {
        Y_ASSERT(IsValid());
        Y_ABORT_UNLESS(SendMMsgFunc, "sendmmsg is not supported!");
        TReadGuard rg(Mutex);
        static bool checked = 0;
        Y_ABORT_UNLESS(checked || (checked = !IsFragmentationForbiden()), "Send methods of this class expect default EnableFragmentation behavior");
        return SendMMsgFunc(S, msgvec, vlen, flags);
    }

    ssize_t TAbstractSocket::SendMsg(const TMsgHdr* hdr, int flags, const EFragFlag frag) {
        Y_ASSERT(IsValid());
#ifdef _win32_
        static bool checked = 0;
        Y_ABORT_UNLESS(hdr->msg_iov->iov_len == 1, "Scatter/gather is currenly not supported on Windows");
        if (hdr->Tos || frag == FF_DONT_FRAG) {
            TWriteGuard wg(Mutex);
            if (frag == FF_DONT_FRAG) {
                ForbidFragmentation();
            } else {
                Y_ABORT_UNLESS(checked || (checked = !IsFragmentationForbiden()), "Send methods of this class expect default EnableFragmentation behavior");
            }
            int originalTos;
            if (hdr->Tos) {
                socklen_t sz = sizeof(originalTos);
                Y_ABORT_UNLESS(GetSockOpt(IPPROTO_IP, IP_TOS, (char*)&originalTos, &sz) == 0, "");
                Y_ABORT_UNLESS(SetSockOpt(IPPROTO_IP, IP_TOS, (char*)&hdr->Tos, sizeof(hdr->Tos)) == 0, "");
            }
            const ssize_t rv = sendto(S, hdr->msg_iov->iov_base, hdr->msg_iov->iov_len, flags, (sockaddr*)hdr->msg_name, hdr->msg_namelen);
            if (hdr->Tos) {
                Y_ABORT_UNLESS(SetSockOpt(IPPROTO_IP, IP_TOS, (char*)&originalTos, sizeof(originalTos)) == 0, "");
            }
            if (frag == FF_DONT_FRAG) {
                EnableFragmentation();
            }
            return rv;
        }
        TReadGuard rg(Mutex);
        Y_ABORT_UNLESS(checked || (checked = !IsFragmentationForbiden()), "Send methods of this class expect default EnableFragmentation behavior");
        return sendto(S, hdr->msg_iov->iov_base, hdr->msg_iov->iov_len, flags, (sockaddr*)hdr->msg_name, hdr->msg_namelen);
#else
        if (frag == FF_DONT_FRAG) {
            TWriteGuard wg(Mutex);
            ForbidFragmentation();
            const ssize_t rv = sendmsg(S, hdr, flags);
            EnableFragmentation();
            return rv;
        }

        TReadGuard rg(Mutex);
#ifndef _darwin_
        static bool checked = 0;
        Y_ABORT_UNLESS(checked || (checked = !IsFragmentationForbiden()), "Send methods of this class expect default EnableFragmentation behavior");
#endif
        return sendmsg(S, hdr, flags);
#endif
    }

    bool TAbstractSocket::IncreaseSendBuff() {
        int buffSize;
        socklen_t sz = sizeof(buffSize);
        if (GetSockOpt(SOL_SOCKET, SO_SNDBUF, &buffSize, &sz)) {
            return false;
        }
        // worst case: 200000 pps * 8k * 0.01sec = 16Mb so 32Mb hard limit is reasonable value
        if (buffSize < 0 || buffSize > (1 << 25)) {
            fprintf(stderr, "GetSockOpt returns wrong or too big value for SO_SNDBUF: %d\n", buffSize);
            return false;
        }
            //linux returns the doubled value. man 7 socket:
            //
            // SO_SNDBUF
            //         Sets or gets the maximum socket send buffer in bytes.  The  ker-
            //         nel doubles this value (to allow space for bookkeeping overhead)
            //         when it is set using setsockopt(), and  this  doubled  value  is
            //         returned  by  getsockopt().   The  default  value  is set by the
            //         wmem_default sysctl and the maximum allowed value is set by  the
            //         wmem_max sysctl.  The minimum (doubled) value for this option is
            //         2048.
            //
#ifndef _linux_
        buffSize += buffSize;
#endif

        // false if previous value was less than current value.
        // It means setsockopt was not successful. (for example: system limits)
        // we will try to set it again but return false
        const bool rv = !(buffSize <= SendSysSocketSizePrev);
        if (SetSockOpt(SOL_SOCKET, SO_SNDBUF, &buffSize, sz) == 0) {
            SendSysSocketSize = buffSize;
            SendSysSocketSizePrev = buffSize;
            return rv;
        }
        return false;
    }

    int TAbstractSocket::GetSendSysSocketSize() {
        return SendSysSocketSize;
    }

    void TAbstractSocket::SetRecvLagTime(NHPTimer::STime time) {
        AtomicSet(RecvLag, time);
    }

    int TAbstractSocket::OpenImpl(int port) {
        Y_ASSERT(!IsValid());
        const int netPort = port ? htons((u_short)port) : 0;

#ifdef _freebsd_
        // alternative OS
        if (netPort == 0) {
            static ui64 pp = GetCycleCount();
            for (int attempt = 0; attempt < 100; ++attempt) {
                const int tryPort = htons((pp & 0x3fff) + 0xc000);
                ++pp;
                if (CreateSocket(tryPort) != 0) {
                    Y_ASSERT(!IsValid());
                    continue;
                }

                if (DetectSelfAddress() != 0 || tryPort != SelfAddress.sin6_port) {
                    // FreeBSD suck!
                    CloseImpl();
                    Y_ASSERT(!IsValid());
                    continue;
                }
                break;
            }
            if (!IsValid()) {
                return -1;
            }
        } else {
            if (CreateSocket(netPort) != 0) {
                Y_ASSERT(!IsValid());
                return -1;
            }
        }
#else
        // regular OS
        if (CreateSocket(netPort) != 0) {
            Y_ASSERT(!IsValid());
            return -1;
        }
#endif

        if (IsValid() && DetectSelfAddress() != 0) {
            CloseImpl();
            Y_ASSERT(!IsValid());
            return -1;
        }

        Y_ASSERT(IsValid());
        return 0;
    }

    void TAbstractSocket::CloseImpl() {
        if (IsValid()) {
            Poller.Unwait(S);
            Y_ABORT_UNLESS(closesocket(S) == 0, "closesocket failed: %s (errno = %d)", LastSystemErrorText(), LastSystemError());
        }
        S = INVALID_SOCKET;
    }

    void TAbstractSocket::WaitImpl(float timeoutSec) const {
        Y_ABORT_UNLESS(IsValid(), "something went wrong");
        Poller.WaitT(TDuration::Seconds(timeoutSec));
    }

    void TAbstractSocket::CancelWaitImpl(const sockaddr_in6* address) {
        Y_ASSERT(IsValid());

        // darwin ignores packets with msg_iovlen == 0, also windows implementation uses sendto of first iovec.
        TIoVec v = CreateIoVec(nullptr, 0);
        TMsgHdr hdr = CreateSendMsgHdr((address ? *address : SelfAddress), v, nullptr);

        // send self fake packet
        TAbstractSocket::SendMsg(&hdr, 0, FF_ALLOW_FRAG);
    }

    ssize_t TAbstractSocket::RecvMsgImpl(TMsgHdr* hdr, int flags) {
        Y_ASSERT(IsValid());

#ifdef _win32_
        Y_ABORT_UNLESS(hdr->msg_iov->iov_len == 1, "Scatter/gather is currenly not supported on Windows");
        return recvfrom(S, hdr->msg_iov->iov_base, hdr->msg_iov->iov_len, flags, (sockaddr*)hdr->msg_name, &hdr->msg_namelen);
#else
        return recvmsg(S, hdr, flags);
#endif
    }

    TUdpRecvPacket* TAbstractSocket::RecvImpl(TUdpHostRecvBufAlloc* buf, sockaddr_in6* srcAddr, sockaddr_in6* dstAddr) {
        Y_ASSERT(IsValid());

        const TIoVec iov = CreateIoVec(buf->GetDataPtr(), buf->GetBufSize());
        char controllBuffer[CTRL_BUFFER_SIZE]; //used to get dst address from socket
        TMsgHdr hdr = CreateRecvMsgHdr(srcAddr, iov, controllBuffer);

        const ssize_t rv = TAbstractSocket::RecvMsgImpl(&hdr, 0);
        if (rv < 0) {
            Y_ASSERT(LastSystemError() == EAGAIN || LastSystemError() == EWOULDBLOCK);
            return nullptr;
        }
        if (dstAddr && !ExtractDestinationAddress(hdr, dstAddr)) {
            //fprintf(stderr, "can`t get destination ip\n");
        }

        // we extract packet and allocate new buffer only if packet arrived
        TUdpRecvPacket* result = buf->ExtractPacket();
        result->DataStart = 0;
        result->DataSize = (int)rv;
        return result;
    }

    // thread-safe
    int TAbstractSocket::RecvMMsgImpl(TMMsgHdr* msgvec, unsigned int vlen, unsigned int flags, timespec* timeout) {
        Y_ASSERT(IsValid());
        Y_ABORT_UNLESS(RecvMMsgFunc, "recvmmsg is not supported!");
        return RecvMMsgFunc(S, msgvec, vlen, flags, timeout);
    }

    ///////////////////////////////////////////////////////////////////////////////

    class TSocket: public TAbstractSocket {
    public:
        int Open(int port) override;
        void Close() override;

        void Wait(float timeoutSec, int netlibaVersion) const override;
        void CancelWait(int netlibaVersion) override;

        bool IsRecvMsgSupported() const override;
        ssize_t RecvMsg(TMsgHdr* hdr, int flags) override;
        TUdpRecvPacket* Recv(sockaddr_in6* srcAddr, sockaddr_in6* dstAddr, int netlibaVersion) override;

    private:
        TUdpHostRecvBufAlloc RecvBuf;
    };

    int TSocket::Open(int port) {
        return OpenImpl(port);
    }

    void TSocket::Close() {
        CloseImpl();
    }

    void TSocket::Wait(float timeoutSec, int netlibaVersion) const {
        Y_UNUSED(netlibaVersion);
        WaitImpl(timeoutSec);
    }

    void TSocket::CancelWait(int netlibaVersion) {
        Y_UNUSED(netlibaVersion);
        CancelWaitImpl();
    }

    bool TSocket::IsRecvMsgSupported() const {
        return true;
    }

    ssize_t TSocket::RecvMsg(TMsgHdr* hdr, int flags) {
        return RecvMsgImpl(hdr, flags);
    }

    TUdpRecvPacket* TSocket::Recv(sockaddr_in6* srcAddr, sockaddr_in6* dstAddr, int netlibaVersion) {
        Y_UNUSED(netlibaVersion);
        return RecvImpl(&RecvBuf, srcAddr, dstAddr);
    }

    ///////////////////////////////////////////////////////////////////////////////

    class TTryToRecvMMsgSocket: public TAbstractSocket {
    private:
        THolderVector<TUdpHostRecvBufAlloc> RecvPackets;
        TVector<sockaddr_in6> RecvPacketsSrcAddresses;
        TVector<TIoVec> RecvPacketsIoVecs;
        size_t RecvPacketsBegin;      // first non returned to user
        size_t RecvPacketsHeadersEnd; // next after last one with data
        TVector<TMMsgHdr> RecvPacketsHeaders;
        TVector<std::array<char, CTRL_BUFFER_SIZE>> RecvPacketsCtrlBuffers;

        int FillRecvBuffers();

    public:
        static bool IsRecvMMsgSupported();

        // Tests showed best performance on queue size 128 (+7%).
        // If memory is limited you can use 12 - it gives +4%.
        // Do not use lower values - for example recvmmsg with 1 element is 3% slower that recvmsg!
        // (tested with junk/f0b0s/neTBasicSocket_queue_test).
        TTryToRecvMMsgSocket(const size_t recvQueueSize = 128);
        ~TTryToRecvMMsgSocket() override;

        int Open(int port) override;
        void Close() override;

        void Wait(float timeoutSec, int netlibaVersion) const override;
        void CancelWait(int netlibaVersion) override;

        bool IsRecvMsgSupported() const override {
            return false;
        }
        ssize_t RecvMsg(TMsgHdr* hdr, int flags) override {
            Y_UNUSED(hdr);
            Y_UNUSED(flags);
            Y_ABORT_UNLESS(false, "Use TBasicSocket for RecvMsg call! TRecvMMsgSocket implementation must use memcpy which is suboptimal and thus forbidden!");
        }
        TUdpRecvPacket* Recv(sockaddr_in6* addr, sockaddr_in6* dstAddr, int netlibaVersion) override;
    };

    TTryToRecvMMsgSocket::TTryToRecvMMsgSocket(const size_t recvQueueSize)
        : RecvPacketsBegin(0)
        , RecvPacketsHeadersEnd(0)
    {
        // recvmmsg is not supported - will act like TSocket,
        // we can't just VERIFY - TTryToRecvMMsgSocket is used as base class for TDualStackSocket.
        if (!IsRecvMMsgSupported()) {
            RecvPackets.reserve(1);
            RecvPackets.PushBack(new TUdpHostRecvBufAlloc);
            return;
        }

        RecvPackets.reserve(recvQueueSize);
        for (size_t i = 0; i != recvQueueSize; ++i) {
            RecvPackets.PushBack(new TUdpHostRecvBufAlloc);
        }

        RecvPacketsSrcAddresses.resize(recvQueueSize);
        RecvPacketsIoVecs.resize(recvQueueSize);
        RecvPacketsHeaders.resize(recvQueueSize);
        RecvPacketsCtrlBuffers.resize(recvQueueSize);

        for (size_t i = 0; i != recvQueueSize; ++i) {
            TMMsgHdr& mhdr = RecvPacketsHeaders[i];
            Zero(mhdr);

            RecvPacketsIoVecs[i] = CreateIoVec(RecvPackets[i]->GetDataPtr(), RecvPackets[i]->GetBufSize());
            char* buf = RecvPacketsCtrlBuffers[i].data();
            memset(buf, 0, CTRL_BUFFER_SIZE);
            mhdr.msg_hdr = CreateRecvMsgHdr(&RecvPacketsSrcAddresses[i], RecvPacketsIoVecs[i], buf);
        }
    }

    TTryToRecvMMsgSocket::~TTryToRecvMMsgSocket() {
        Close();
    }

    int TTryToRecvMMsgSocket::Open(int port) {
        return OpenImpl(port);
    }

    void TTryToRecvMMsgSocket::Close() {
        CloseImpl();
    }

    void TTryToRecvMMsgSocket::Wait(float timeoutSec, int netlibaVersion) const {
        Y_UNUSED(netlibaVersion);
        Y_ASSERT(RecvPacketsBegin == RecvPacketsHeadersEnd || IsRecvMMsgSupported());
        if (RecvPacketsBegin == RecvPacketsHeadersEnd) {
            WaitImpl(timeoutSec);
        }
    }

    void TTryToRecvMMsgSocket::CancelWait(int netlibaVersion) {
        Y_UNUSED(netlibaVersion);
        CancelWaitImpl();
    }

    bool TTryToRecvMMsgSocket::IsRecvMMsgSupported() {
        return RecvMMsgFunc != nullptr;
    }

    int TTryToRecvMMsgSocket::FillRecvBuffers() {
        Y_ASSERT(IsRecvMMsgSupported());
        Y_ASSERT(RecvPacketsBegin <= RecvPacketsHeadersEnd);
        if (RecvPacketsBegin < RecvPacketsHeadersEnd) {
            return RecvPacketsHeadersEnd - RecvPacketsBegin;
        }

        // no packets left from last recvmmsg call
        for (size_t i = 0; i != RecvPacketsHeadersEnd; ++i) { // reinit only used by last recvmmsg call headers
            RecvPacketsIoVecs[i] = CreateIoVec(RecvPackets[i]->GetDataPtr(), RecvPackets[i]->GetBufSize());
        }
        RecvPacketsBegin = RecvPacketsHeadersEnd = 0;

        const int r = RecvMMsgImpl(&RecvPacketsHeaders[0], (unsigned int)RecvPacketsHeaders.size(), 0, nullptr);
        if (r >= 0) {
            RecvPacketsHeadersEnd = r;
        } else {
            Y_ASSERT(LastSystemError() == EAGAIN || LastSystemError() == EWOULDBLOCK);
        }
        return r;
    }

    // not thread-safe
    TUdpRecvPacket* TTryToRecvMMsgSocket::Recv(sockaddr_in6* fromAddress, sockaddr_in6* dstAddr, int) {
        // act like TSocket
        if (!IsRecvMMsgSupported()) {
            return RecvImpl(RecvPackets[0], fromAddress, dstAddr);
        }

        if (FillRecvBuffers() <= 0) {
            return nullptr;
        }

        TUdpRecvPacket* result = RecvPackets[RecvPacketsBegin]->ExtractPacket();
        TMMsgHdr& mmsgHdr = RecvPacketsHeaders[RecvPacketsBegin];
        result->DataSize = (ssize_t)mmsgHdr.msg_len;
        if (dstAddr && !ExtractDestinationAddress(mmsgHdr.msg_hdr, dstAddr)) {
            //    fprintf(stderr, "can`t get destination ip\n");
        }
        *fromAddress = RecvPacketsSrcAddresses[RecvPacketsBegin];
        //we must clean ctrlbuffer to be able to use it later
#ifndef _win_
        memset(mmsgHdr.msg_hdr.msg_control, 0, CTRL_BUFFER_SIZE);
        mmsgHdr.msg_hdr.msg_controllen = CTRL_BUFFER_SIZE;
#endif
        RecvPacketsBegin++;

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////

    /*  TODO: too slow, needs to be optimized
template<size_t TTNumRecvThreads>
class TMTRecvSocket: public TAbstractSocket
{
private:
    typedef TLockFreePacketQueue<TTNumRecvThreads> TPacketQueue;

    static void* RecvThreadFunc(void* that)
    {
        static_cast<TMTRecvSocket*>(that)->RecvLoop();
        return NULL;
    }

    void RecvLoop()
    {
        TBestUnixRecvSocket impl;
        impl.Reset(*this);

        while (AtomicAdd(NumThreadsToDie, 0) == -1) {
            sockaddr_in6 addr;
            TUdpRecvPacket* packet = impl.Recv(&addr, NETLIBA_ANY_VERSION);
            if (!packet) {
                impl.Wait(0.0001, NETLIBA_ANY_VERSION);  // so small tiomeout because we can't guarantee that 1 thread won't get all packets
                continue;
            }
            Queue.Push(packet, addr);
        }

        if (AtomicDecrement(NumThreadsToDie)) {
            impl.CancelWait(NETLIBA_ANY_VERSION);
        } else {
            AllThreadsAreDead.Signal();
        }
    }

    THolderVector<TThread> RecvThreads;
    TAtomic NumThreadsToDie;
    TSystemEvent AllThreadsAreDead;

    TPacketQueue Queue;

public:
    TMTRecvSocket()
        : NumThreadsToDie(-1) {}

    ~TMTRecvSocket()
    {
        Close();
    }

    int Open(int port)
    {
        if (OpenImpl(port) != 0) {
            Y_ASSERT(!IsValid());
            return -1;
        }

        NumThreadsToDie = -1;
        RecvThreads.reserve(TTNumRecvThreads);
        for (size_t i = 0; i != TTNumRecvThreads; ++i) {
            RecvThreads.PushBack(new TThread(TThread::TParams(RecvThreadFunc, this).SetName("nl12_recv_skt")));
            RecvThreads.back()->Start();
            RecvThreads.back()->Detach();
        }
        return 0;
    }

    void Close()
    {
        if (!IsValid()) {
            return;
        }

        AtomicSwap(&NumThreadsToDie, (int)RecvThreads.size());
        CancelWaitImpl();
        Y_ABORT_UNLESS(AllThreadsAreDead.WaitT(TDuration::Seconds(30)), "TMTRecvSocket destruction failed");

        CloseImpl();
    }

    void Wait(float timeoutSec, int netlibaVersion) const
    {
        Y_UNUSED(netlibaVersion);
        Queue.GetEvent().WaitT(TDuration::Seconds(timeoutSec));
    }
    void CancelWait(int netlibaVersion)
    {
        Y_UNUSED(netlibaVersion);
        Queue.GetEvent().Signal();
    }

    TUdpRecvPacket* Recv(sockaddr_in6 *addr, int netlibaVersion)
    {
        Y_UNUSED(netlibaVersion);
        TUdpRecvPacket* result;
        if (!Queue.Pop(&result, addr)) {
            return NULL;
        }
        return result;
    }

    bool IsRecvMsgSupported() const { return false; }
    ssize_t RecvMsg(TMsgHdr* hdr, int flags) { Y_ABORT_UNLESS(false, "Use TBasicSocket for RecvMsg call! TMTRecvSocket implementation must use memcpy which is suboptimal and thus forbidden!"); }
};
*/

    ///////////////////////////////////////////////////////////////////////////////

    // Send.*, Recv, Wait and CancelWait are thread-safe.
    class TDualStackSocket: public TTryToRecvMMsgSocket {
    private:
        typedef TTryToRecvMMsgSocket TBase;
        typedef TLockFreePacketQueue<1> TPacketQueue;

        static void* RecvThreadFunc(void* that);
        void RecvLoop();

        struct TFilteredPacketQueue {
            enum EPushResult {
                PR_FULL = 0,
                PR_OK = 1,
                PR_FILTERED = 2
            };
            const ui8 F1;
            const ui8 F2;
            const ui8 CmdPos;
            TFilteredPacketQueue(ui8 f1, ui8 f2, ui8 cmdPos)
                : F1(f1)
                , F2(f2)
                , CmdPos(cmdPos)
            {
            }
            bool Pop(TUdpRecvPacket** packet, sockaddr_in6* srcAddr, sockaddr_in6* dstAddr) {
                return Queue.Pop(packet, srcAddr, dstAddr);
            }
            ui8 Push(TUdpRecvPacket* packet, const TPacketMeta& meta) {
                if (Queue.IsDataPartFull()) {
                    const ui8 cmd = packet->Data.get()[CmdPos];
                    if (cmd == F1 || cmd == F2)
                        return PR_FILTERED;
                }
                return Queue.Push(packet, meta); //false - PR_FULL, true - PR_OK
            }
            TPacketQueue Queue;
        };

        TFilteredPacketQueue& GetRecvQueue(int netlibaVersion) const;
        TSystemEvent& GetQueueEvent(const TFilteredPacketQueue& queue) const;

        TThread RecvThread;
        TAtomic ShouldDie;
        TSystemEvent DieEvent;

        mutable TFilteredPacketQueue RecvQueue6;
        mutable TFilteredPacketQueue RecvQueue12;

    public:
        TDualStackSocket();
        ~TDualStackSocket() override;

        int Open(int port) override;
        void Close() override;

        void Wait(float timeoutSec, int netlibaVersion) const override;
        void CancelWait(int netlibaVersion) override;

        bool IsRecvMsgSupported() const override {
            return false;
        }
        ssize_t RecvMsg(TMsgHdr* hdr, int flags) override {
            Y_UNUSED(hdr);
            Y_UNUSED(flags);
            Y_ABORT_UNLESS(false, "Use TBasicSocket for RecvMsg call! TDualStackSocket implementation must use memcpy which is suboptimal and thus forbidden!");
        }

        TUdpRecvPacket* Recv(sockaddr_in6* addr, sockaddr_in6* dstAddr, int netlibaVersion) override;
    };

    TDualStackSocket::TDualStackSocket()
        : RecvThread(TThread::TParams(RecvThreadFunc, this).SetName("nl12_dual_stack"))
        , ShouldDie(0)
        , RecvQueue6(NNetliba::DATA, NNetliba::DATA_SMALL, NNetliba::CMD_POS)
        , RecvQueue12(NNetliba_v12::DATA, NNetliba_v12::DATA_SMALL, NNetliba_v12::CMD_POS)
    {
    }

    // virtual functions don't work in dtors!
    TDualStackSocket::~TDualStackSocket() {
        Close();

        sockaddr_in6 srcAdd;
        sockaddr_in6 dstAddr;
        TUdpRecvPacket* ptr = nullptr;

        while (GetRecvQueue(NETLIBA_ANY_VERSION).Pop(&ptr, &srcAdd, &dstAddr)) {
            delete ptr;
        }
        while (GetRecvQueue(NETLIBA_V12_VERSION).Pop(&ptr, &srcAdd, &dstAddr)) {
            delete ptr;
        }
    }

    int TDualStackSocket::Open(int port) {
        if (TBase::Open(port) != 0) {
            Y_ASSERT(!IsValid());
            return -1;
        }

        AtomicSet(ShouldDie, 0);
        DieEvent.Reset();
        RecvThread.Start();
        RecvThread.Detach();
        return 0;
    }

    void TDualStackSocket::Close() {
        if (!IsValid()) {
            return;
        }

        AtomicSwap(&ShouldDie, 1);
        CancelWaitImpl();
        Y_ABORT_UNLESS(DieEvent.WaitT(TDuration::Seconds(30)), "TDualStackSocket::Close failed");

        TBase::Close();
    }

    TDualStackSocket::TFilteredPacketQueue& TDualStackSocket::GetRecvQueue(int netlibaVersion) const {
        return netlibaVersion == NETLIBA_V12_VERSION ? RecvQueue12 : RecvQueue6;
    }

    TSystemEvent& TDualStackSocket::GetQueueEvent(const TFilteredPacketQueue& queue) const {
        return queue.Queue.GetEvent();
    }

    void* TDualStackSocket::RecvThreadFunc(void* that) {
        SetHighestThreadPriority();
        static_cast<TDualStackSocket*>(that)->RecvLoop();
        return nullptr;
    }

    void TDualStackSocket::RecvLoop() {
        for (;;) {
            TUdpRecvPacket* p = nullptr;
            sockaddr_in6 srcAddr;
            sockaddr_in6 dstAddr;
            while (AtomicAdd(ShouldDie, 0) == 0 && (p = TBase::Recv(&srcAddr, &dstAddr, NETLIBA_ANY_VERSION))) {
                Y_ASSERT(p->DataStart == 0);
                if (p->DataSize < 12) {
                    continue;
                }

                TFilteredPacketQueue& q = GetRecvQueue(p->Data.get()[8]);
                const ui8 res = q.Push(p, {srcAddr, dstAddr});
                if (res == TFilteredPacketQueue::PR_OK) {
                    GetQueueEvent(q).Signal();
                } else {
                    // simulate OS behavior on buffer overflow - drop packets.
                    const NHPTimer::STime time = AtomicGet(RecvLag);
                    const float sec = NHPTimer::GetSeconds(time);
                    fprintf(stderr, "TDualStackSocket::RecvLoop netliba v%d queue overflow, recv lag: %f sec, dropping packet, res: %u\n",
                            &q == &RecvQueue12 ? 12 : 6, sec, res);
                    delete p;
                }
            }

            if (AtomicAdd(ShouldDie, 0)) {
                DieEvent.Signal();
                return;
            }

            TBase::Wait(0.1f, NETLIBA_ANY_VERSION);
        }
    }

    void TDualStackSocket::Wait(float timeoutSec, int netlibaVersion) const {
        TFilteredPacketQueue& q = GetRecvQueue(netlibaVersion);
        if (q.Queue.IsEmpty()) {
            GetQueueEvent(q).Reset();
            if (q.Queue.IsEmpty()) {
                GetQueueEvent(q).WaitT(TDuration::Seconds(timeoutSec));
            }
        }
    }

    void TDualStackSocket::CancelWait(int netlibaVersion) {
        GetQueueEvent(GetRecvQueue(netlibaVersion)).Signal();
    }

    // thread-safe
    TUdpRecvPacket* TDualStackSocket::Recv(sockaddr_in6* srcAddr, sockaddr_in6* dstAddr, int netlibaVersion) {
        TUdpRecvPacket* result = nullptr;
        if (!GetRecvQueue(netlibaVersion).Pop(&result, srcAddr, dstAddr)) {
            return nullptr;
        }
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////////

    TIntrusivePtr<ISocket> CreateSocket() {
        return new TSocket();
    }

    TIntrusivePtr<ISocket> CreateDualStackSocket() {
        return new TDualStackSocket();
    }

    TIntrusivePtr<ISocket> CreateBestRecvSocket() {
        // TSocket is faster than TRecvMMsgFunc in case of unsupported recvmmsg
        if (!TTryToRecvMMsgSocket::IsRecvMMsgSupported()) {
            return new TSocket();
        }
        return new TTryToRecvMMsgSocket();
    }

}
