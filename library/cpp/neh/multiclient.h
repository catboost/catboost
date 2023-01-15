#pragma once

#include "neh.h"

namespace NNeh {
    /// thread-safe dispacher for processing multiple neh requests
    /// (method Wait() MUST be called from single thread, methods Request and Interrupt are thread-safe)
    class IMultiClient {
    public:
        virtual ~IMultiClient() {
        }

        struct TRequest {
            TRequest()
                : Deadline(TInstant::Max())
                , UserData(nullptr)
            {
            }

            TRequest(const TMessage& msg, TInstant deadline = TInstant::Max(), void* userData = nullptr)
                : Msg(msg)
                , Deadline(deadline)
                , UserData(userData)
            {
            }

            TMessage Msg;
            TInstant Deadline;
            void* UserData;
        };

        /// WARNING:
        ///  Wait(event) called from another thread can return Event
        ///  for this request before this call return control
        virtual THandleRef Request(const TRequest& req) = 0;

        virtual size_t QueueSize() = 0;

        struct TEvent {
            enum TType {
                Timeout,
                Response,
                SizeEventType
            };

            TEvent()
                : Type(SizeEventType)
                , UserData(nullptr)
            {
            }

            TEvent(TType t, void* userData)
                : Type(t)
                , UserData(userData)
            {
            }

            TType Type;
            THandleRef Hndl;
            void* UserData;
        };

        /// return false if interrupted
        virtual bool Wait(TEvent&, TInstant = TInstant::Max()) = 0;
        /// interrupt guaranteed breaking execution Wait(), but few interrupts can be handled as one
        virtual void Interrupt() = 0;
    };

    typedef TAutoPtr<IMultiClient> TMultiClientPtr;

    TMultiClientPtr CreateMultiClient();
}
