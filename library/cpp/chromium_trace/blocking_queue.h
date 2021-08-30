#pragma once

#include "samplers.h"
#include "tracer.h"

#include <util/generic/ptr.h>
#include <util/system/mutex.h>

namespace NChromiumTrace {
    template <typename TBaseBlockingQueue>
    class TTraceBlockingQueue: public TBaseBlockingQueue {
        class TObserver final: public TSharedSamplerBase {
            TMutex Lock;
            const TTraceBlockingQueue* Queue;

        public:
            TObserver(const TTraceBlockingQueue* queue)
                : Queue(queue)
            {
            }

            void Disconnect() {
                with_lock (Lock) {
                    Queue = nullptr;
                }
            }

            void Update() override {
            }

            void Publish(TTracer& tracer) const override {
                with_lock (Lock) {
                    if (!Queue) {
                        return;
                    }

                    tracer.AddCounterNow(
                        Queue->Name,
                        TStringBuf("sample"),
                        TEventArgs().Add(TStringBuf("Size"), (i64)Queue->Size()));
                }
            }
        };

        TString Name;
        TIntrusivePtr<TObserver> Observer;

    public:
        template <typename... TArgs>
        TTraceBlockingQueue(const TString& name, TArgs&&... args)
            : TBaseBlockingQueue(std::forward<TArgs>(args)...)
            , Name(name)
        {
        }

        ~TTraceBlockingQueue() {
            if (Observer) {
                Observer->Disconnect();
            }
        }

        TProxySampler GetSampler() {
            if (!Observer) {
                Observer = MakeIntrusive<TObserver>(this);
            }

            return {Observer};
        }
    };
}
