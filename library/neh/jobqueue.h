#pragma once

#include <library/cpp/coroutine/engine/impl.h>

#include <util/generic/yexception.h>
#include <util/stream/output.h>

namespace NNeh {
    class IJob {
    public:
        inline void operator()(TCont* c) noexcept {
            try {
                DoRun(c);
            } catch (...) {
                Cdbg << "run " << CurrentExceptionMessage() << Endl;
            }
        }

        virtual ~IJob() {
        }

    private:
        virtual void DoRun(TCont* c) = 0;
    };

    class IJobQueue {
    public:
        template <class T>
        inline void Schedule(T req) {
            ScheduleImpl(req.Get());
            Y_UNUSED(req.Release());
        }

        virtual void ScheduleImpl(IJob* job) = 0;

        virtual ~IJobQueue() {
        }
    };

    IJobQueue* JobQueue();
}
