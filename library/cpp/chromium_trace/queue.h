#pragma once

#include "global.h"

#include <util/generic/string.h>
#include <util/string/builder.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/thread/pool.h>

namespace NChromiumTrace {
    template <typename TBaseMtpQueue>
    class TTraceMtpQueue: public TBaseMtpQueue {
        TString TraceName;
        TAtomic ThreadCounter;

    public:
        template <typename... Args>
        TTraceMtpQueue(Args&&... args)
            : TBaseMtpQueue(std::forward<Args>(args)...)
            , TraceName(TStringBuilder() << "Pool@" << (void*)this)
            , ThreadCounter(0)
        {
        }

        void SetTraceName(const TString& name) {
            TraceName = name;
        }

        const TString& GetTraceName() const {
            return TraceName;
        }

    private:
        void* CreateThreadSpecificResource() override {
            auto threadId = AtomicAdd(ThreadCounter, 1);
            GetGlobalTracer()->AddCurrentThreadName(
                TStringBuilder() << TraceName << ':' << threadId);
            return TBaseMtpQueue::CreateThreadSpecificResource();
        }

        void DestroyThreadSpecificResource(void* resource) override {
            TBaseMtpQueue::DestroyThreadSpecificResource(resource);
        }
    };

}
