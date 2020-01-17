#include "stdafx.h"
#include <string.h>
#include <util/generic/utility.h>
#include <util/network/init.h>
#include <util/system/defaults.h>
#include <util/system/yassert.h>
#include "socket.h"

namespace NNetlibaSocket {
    void* CreateTos(const ui8 tos, void* buffer) {
#ifdef _win_
        *(int*)buffer = (int)tos;
#else
        // glibc bug: http://sourceware.org/bugzilla/show_bug.cgi?id=13500
        memset(buffer, 0, TOS_BUFFER_SIZE);

        msghdr dummy;
        Zero(dummy);
        dummy.msg_control = buffer;
        dummy.msg_controllen = TOS_BUFFER_SIZE;

        // TODO: in FreeBSD setting TOS for dual stack sockets does not affect ipv4 frames
        cmsghdr* cmsg = CMSG_FIRSTHDR(&dummy);
        cmsg->cmsg_level = IPPROTO_IPV6;
        cmsg->cmsg_type = IPV6_TCLASS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cmsg), &tos, sizeof(tos)); // memcpy shut ups alias restrict warning

        Y_ASSERT(CMSG_NXTHDR(&dummy, cmsg) == nullptr);
#endif
        return buffer;
    }

    TMsgHdr* AddSockAuxData(TMsgHdr* header, const ui8 tos, const sockaddr_in6& myAddr, void* buffer, size_t bufferSize) {
#ifdef _win_
        Y_UNUSED(header);
        Y_UNUSED(tos);
        Y_UNUSED(myAddr);
        Y_UNUSED(buffer);
        Y_UNUSED(bufferSize);
        return nullptr;
#else
        header->msg_control = buffer;
        header->msg_controllen = bufferSize;

        size_t totalLen = 0;
#ifdef _cygwin_
        Y_UNUSED(tos);
#else
        // Cygwin does not support IPV6_TCLASS, so we ignore it
        cmsghdr* cmsgTos = CMSG_FIRSTHDR(header);
        if (cmsgTos == nullptr) {
            header->msg_control = nullptr;
            header->msg_controllen = 0;
            return nullptr;
        }
        cmsgTos->cmsg_level = IPPROTO_IPV6;
        cmsgTos->cmsg_type = IPV6_TCLASS;
        cmsgTos->cmsg_len = CMSG_LEN(sizeof(int));
        totalLen = CMSG_SPACE(sizeof(int));
        *(ui8*)CMSG_DATA(cmsgTos) = tos;
#endif

        if (*(ui64*)myAddr.sin6_addr.s6_addr != 0u) {
            in6_pktinfo* pktInfo;
#ifdef _cygwin_
            cmsghdr* cmsgAddr = CMSG_FIRSTHDR(header);
#else
            cmsghdr* cmsgAddr = CMSG_NXTHDR(header, cmsgTos);
#endif
            if (cmsgAddr == nullptr) {
                // leave only previous record
                header->msg_controllen = totalLen;
                return nullptr;
            }
            cmsgAddr->cmsg_level = IPPROTO_IPV6;
            cmsgAddr->cmsg_type = IPV6_PKTINFO;
            cmsgAddr->cmsg_len = CMSG_LEN(sizeof(*pktInfo));
            totalLen += CMSG_SPACE(sizeof(*pktInfo));
            pktInfo = (in6_pktinfo*)CMSG_DATA(cmsgAddr);

            pktInfo->ipi6_addr = myAddr.sin6_addr;
            pktInfo->ipi6_ifindex = 0; /* 0 = use interface specified in routing table */
        }
        header->msg_controllen = totalLen; //write right len

        return header;
#endif
    }

    TIoVec CreateIoVec(char* data, const size_t dataSize) {
        TIoVec result;
        Zero(result);

        result.iov_base = data;
        result.iov_len = dataSize;

        return result;
    }

    TMsgHdr CreateSendMsgHdr(const sockaddr_in6& addr, const TIoVec& iov, void* tosBuffer) {
        TMsgHdr result;
        Zero(result);

        result.msg_name = (void*)&addr;
        result.msg_namelen = sizeof(addr);
        result.msg_iov = (TIoVec*)&iov;
        result.msg_iovlen = 1;

        if (tosBuffer) {
#ifdef _win_
            result.Tos = *(int*)tosBuffer;
#else
            result.msg_control = tosBuffer;
            result.msg_controllen = TOS_BUFFER_SIZE;
#endif
        }

        return result;
    }

    TMsgHdr CreateRecvMsgHdr(sockaddr_in6* addrBuf, const TIoVec& iov, void* controllBuffer) {
        TMsgHdr result;
        Zero(result);

        Zero(*addrBuf);
        result.msg_name = addrBuf;
        result.msg_namelen = sizeof(*addrBuf);

        result.msg_iov = (TIoVec*)&iov;
        result.msg_iovlen = 1;
#ifndef _win_
        if (controllBuffer) {
            memset(controllBuffer, 0, CTRL_BUFFER_SIZE);
            result.msg_control = controllBuffer;
            result.msg_controllen = CTRL_BUFFER_SIZE;
        }
#endif
        return result;
    }
}
