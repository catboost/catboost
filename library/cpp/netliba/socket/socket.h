#pragma once

#include <util/system/platform.h>
#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/network/init.h>
#include <util/system/defaults.h>
#include <util/system/hp_timer.h>
#include "udp_recv_packet.h"
#include "protocols.h"

#include <sys/uio.h>

namespace NNetlibaSocket {
    typedef iovec TIoVec;

#ifdef _win32_
   struct TMsgHdr {
        void* msg_name;  /* optional address */
        int msg_namelen; /* size of address */
        TIoVec* msg_iov; /* scatter/gather array */
        int msg_iovlen;  /* # elements in msg_iov */

        int Tos; // netlib_socket extension
    };
#else
#include <sys/socket.h>
    typedef msghdr TMsgHdr;
#endif

    // equal to glibc 2.14 mmsghdr definition, defined for windows and darwin compatibility
    struct TMMsgHdr {
        TMsgHdr msg_hdr;
        unsigned int msg_len;
    };

#if defined(_linux_)
#include <linux/version.h>
#include <features.h>
// sendmmsg was added in glibc 2.14 and linux 3.0
#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 14 && LINUX_VERSION_CODE >= KERNEL_VERSION(3, 0, 0)
#include <sys/socket.h> // sendmmsg
    static_assert(sizeof(TMMsgHdr) == sizeof(mmsghdr), "expect sizeof(TMMsgHdr) == sizeof(mmsghdr)");
#endif
#endif

#ifdef _win32_
    const size_t TOS_BUFFER_SIZE = sizeof(int);
    const size_t CTRL_BUFFER_SIZE = 32;
#else
#if defined(_darwin_)
#define Y_DARWIN_ALIGN32(p) ((__darwin_size_t)((__darwin_size_t)(p) + __DARWIN_ALIGNBYTES32) & ~__DARWIN_ALIGNBYTES32)
#define Y_CMSG_SPACE(l) (Y_DARWIN_ALIGN32(sizeof(struct cmsghdr)) + Y_DARWIN_ALIGN32(l))
#else
#define Y_CMSG_SPACE(l) CMSG_SPACE(l)
#endif

    constexpr size_t TOS_BUFFER_SIZE = Y_CMSG_SPACE(sizeof(int));
    constexpr size_t CTRL_BUFFER_SIZE = Y_CMSG_SPACE(sizeof(int)) + Y_CMSG_SPACE(sizeof(struct in6_pktinfo));
#endif

    ///////////////////////////////////////////////////////////////////////////////
    // Warning: every variable (tosBuffer, data, addr, iov) passed and returned from these functions must exist until actual send!!!
    void* CreateTos(const ui8 tos, void* tosBuffer);
    TIoVec CreateIoVec(char* data, const size_t dataSize);
    TMsgHdr CreateSendMsgHdr(const sockaddr_in6& addr, const TIoVec& iov, void* tosBuffer);
    TMsgHdr CreateRecvMsgHdr(sockaddr_in6* addrBuf, const TIoVec& iov, void* ctrlBuffer = nullptr);
    TMsgHdr* AddSockAuxData(TMsgHdr* header, const ui8 tos, const sockaddr_in6& addr, void* buffer, size_t bufferSize);
    ///////////////////////////////////////////////////////////////////////////////
    //returns false if TOS wasn't readed and do not touch *tos
    bool ReadTos(const TMsgHdr& msgHdr, ui8* tos);
    bool ExtractDestinationAddress(TMsgHdr& msgHdr, sockaddr_in6* addrBuf);

    ///////////////////////////////////////////////////////////////////////////////

    // currently netliba v6 version id is any number which's not equal to NETLIBA_V12_VERSION
    constexpr int NETLIBA_ANY_VERSION = -1;
    constexpr int NETLIBA_V12_VERSION = 112;

    enum EFragFlag {
        FF_ALLOW_FRAG,
        FF_DONT_FRAG
    };

    ///////////////////////////////////////////////////////////////////////////////

    class ISocket: public TNonCopyable, public TThrRefBase {
    public:
        ~ISocket() override {
        }

        virtual int Open(int port) = 0;
        virtual void Close() = 0;
        virtual bool IsValid() const = 0;

        virtual const sockaddr_in6& GetSelfAddress() const = 0;
        virtual int GetNetworkOrderPort() const = 0;
        virtual int GetPort() const = 0;

        virtual int GetSockOpt(int level, int option_name, void* option_value, socklen_t* option_len) = 0;

        // send all packets to this and only this address by default
        virtual int Connect(const struct sockaddr* address, socklen_t address_len) = 0;

        virtual void Wait(float timeoutSec, int netlibaVersion = NETLIBA_ANY_VERSION) const = 0;
        virtual void CancelWait(int netlibaVersion = NETLIBA_ANY_VERSION) = 0;
        virtual void CancelWaitHost(const sockaddr_in6 address) = 0;

        virtual bool IsSendMMsgSupported() const = 0;
        virtual int SendMMsg(struct TMMsgHdr* msgvec, unsigned int vlen, unsigned int flags) = 0;
        virtual ssize_t SendMsg(const TMsgHdr* hdr, int flags, const EFragFlag frag) = 0;

        virtual bool IsRecvMsgSupported() const = 0;
        virtual ssize_t RecvMsg(TMsgHdr* hdr, int flags) = 0;
        virtual TUdpRecvPacket* Recv(sockaddr_in6* srcAddr, sockaddr_in6* dstAddr, int netlibaVersion = NETLIBA_ANY_VERSION) = 0;
        virtual bool IncreaseSendBuff() = 0;
        virtual int GetSendSysSocketSize() = 0;
        virtual void SetRecvLagTime(NHPTimer::STime time) = 0;
    };

    TIntrusivePtr<ISocket> CreateSocket();          // not thread safe!
    TIntrusivePtr<ISocket> CreateDualStackSocket(); // has thread safe send/recv methods

    // this function was added mostly for testing
    TIntrusivePtr<ISocket> CreateBestRecvSocket();
}
