#pragma once

#include <util/system/defaults.h>
#include <util/generic/yexception.h>
#include <util/network/socket.h>
#include <util/system/pipe.h>

#ifdef _linux_
#include <sys/eventfd.h>
#endif

#if defined(_bionic_) && !defined(EFD_SEMAPHORE)
#define EFD_SEMAPHORE 1
#endif

namespace NAsio {
#ifdef _linux_
    class TEventFdPollInterrupter {
    public:
        inline TEventFdPollInterrupter() {
            F_ = eventfd(0, EFD_NONBLOCK | EFD_SEMAPHORE);
            if (F_ < 0) {
                ythrow TFileError() << "failed to create a eventfd";
            }
        }

        inline ~TEventFdPollInterrupter() {
            close(F_);
        }

        inline void Interrupt() const noexcept {
            const static eventfd_t ev(1);
            ssize_t res = ::write(F_, &ev, sizeof ev);
            Y_UNUSED(res);
        }

        inline bool Reset() const noexcept {
            eventfd_t ev(0);

            for (;;) {
                ssize_t res = ::read(F_, &ev, sizeof ev);
                if (res && res == EINTR) {
                    continue;
                }

                return res > 0;
            }
        }

        int Fd() {
            return F_;
        }

    private:
        int F_;
    };
#endif

    class TPipePollInterrupter {
    public:
        TPipePollInterrupter() {
            TPipeHandle::Pipe(S_[0], S_[1]);

            SetNonBlock(S_[0]);
            SetNonBlock(S_[1]);
        }

        inline void Interrupt() const noexcept {
            char byte = 0;
            ssize_t res = S_[1].Write(&byte, 1);
            Y_UNUSED(res);
        }

        inline bool Reset() const noexcept {
            char buff[256];

            for (;;) {
                ssize_t r = S_[0].Read(buff, sizeof buff);

                if (r < 0 && r == EINTR) {
                    continue;
                }

                bool wasInterrupted = r > 0;

                while (r == sizeof buff) {
                    r = S_[0].Read(buff, sizeof buff);
                }

                return wasInterrupted;
            }
        }

        PIPEHANDLE Fd() const noexcept {
            return S_[0];
        }

    private:
        TPipeHandle S_[2];
    };

#ifdef _linux_
    typedef TEventFdPollInterrupter TPollInterrupter; //more effective than pipe, but only linux impl.
#else
    typedef TPipePollInterrupter TPollInterrupter;
#endif
}
