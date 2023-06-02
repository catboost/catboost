#include "notification_handle.h"

#include <library/cpp/yt/system/handle_eintr.h>

#include <library/cpp/yt/assert/assert.h>

#ifdef _linux_
    #include <unistd.h>
    #include <sys/eventfd.h>
#endif

#ifdef _darwin_
    #include <fcntl.h>
    #include <unistd.h>
#endif

#ifdef _win_
    #include <util/network/socket.h>
#endif

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

TNotificationHandle::TNotificationHandle(bool blocking)
{
#ifdef _linux_
    EventFD_ = HandleEintr(
        eventfd,
        0,
        EFD_CLOEXEC | (blocking ? 0 : EFD_NONBLOCK));
    YT_VERIFY(EventFD_ >= 0);
#elif defined(_win_)
    TPipeHandle::Pipe(Reader_, Writer_, EOpenModeFlag::CloseOnExec);
    if (!blocking) {
        SetNonBlock(Reader_);
    }
#else
#ifdef _darwin_
    YT_VERIFY(HandleEintr(pipe, PipeFDs_) == 0);
#else
    YT_VERIFY(HandleEintr(pipe2, PipeFDs_, O_CLOEXEC) == 0);
#endif
    if (!blocking) {
        YT_VERIFY(fcntl(PipeFDs_[0], F_SETFL, O_NONBLOCK) == 0);
    }
#endif
}

TNotificationHandle::~TNotificationHandle()
{
#ifdef _linux_
    YT_VERIFY(HandleEintr(close, EventFD_) == 0);
#elif !defined(_win_)
    YT_VERIFY(HandleEintr(close, PipeFDs_[0]) == 0);
    YT_VERIFY(HandleEintr(close, PipeFDs_[1]) == 0);
#endif
}

void TNotificationHandle::Raise()
{
#ifdef _linux_
    uint64_t one = 1;
    YT_VERIFY(HandleEintr(write, EventFD_, &one, sizeof(one)) == sizeof(one));
#elif defined(_win_)
    char c = 'x';
    YT_VERIFY(Writer_.Write(&c, sizeof(char)) == sizeof(char));
#else
    char c = 'x';
    YT_VERIFY(HandleEintr(write, PipeFDs_[1], &c, sizeof(char)) == sizeof(char));
#endif
}

void TNotificationHandle::Clear()
{
#ifdef _linux_
    uint64_t count = 0;
    auto ret = HandleEintr(read, EventFD_, &count, sizeof(count));
    // For edge-triggered one could clear multiple events, others get nothing.
    YT_VERIFY(ret == sizeof(count) || (ret < 0 && errno == EAGAIN));
#elif defined(_win_)
    while (true) {
        char c;
        auto ret = Reader_.Read(&c, sizeof(c));
        YT_VERIFY(ret == sizeof(c) || (ret == SOCKET_ERROR && WSAGetLastError() == WSAEWOULDBLOCK));
        if (ret == SOCKET_ERROR) {
            break;
        }
    }
#else
    while (true) {
        char c;
        auto ret = HandleEintr(read, PipeFDs_[0], &c, sizeof(c));
        YT_VERIFY(ret == sizeof(c) || (ret < 0 && errno == EAGAIN));
        if (ret < 0) {
            break;
        }
    }
#endif
}

int TNotificationHandle::GetFD() const
{
#ifdef _linux_
    return EventFD_;
#elif defined(_win_)
    return Reader_;
#else
    return PipeFDs_[0];
#endif
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
