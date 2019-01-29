#pragma once

#include "fwd.h"
#include "future.h"

#include <util/thread/factory.h>

#include <functional>

namespace NThreading {
    template <typename TR, bool IgnoreException>
    class TLegacyFuture: public IThreadFactory::IThreadAble, TNonCopyable {
    public:
        typedef TR(TFunctionSignature)();
        using TFunctionObjectType = std::function<TFunctionSignature>;
        using TResult = typename TFunctionObjectType::result_type;

    private:
        TFunctionObjectType Func_;
        TPromise<TResult> Result_;
        THolder<IThreadFactory::IThread> Thread_;

    public:
        inline TLegacyFuture(const TFunctionObjectType func, IThreadFactory* pool = SystemThreadFactory())
            : Func_(func)
            , Result_(NewPromise<TResult>())
            , Thread_(pool->Run(this))
        {
        }

        inline ~TLegacyFuture() override {
            this->Join();
        }

        inline TResult Get() {
            this->Join();
            return Result_.GetValue();
        }

    private:
        inline void Join() {
            if (Thread_) {
                Thread_->Join();
                Thread_.Destroy();
            }
        }

        template <typename Result, bool IgnoreException_>
        struct TExecutor {
            static void SetPromise(TPromise<Result>& promise, const TFunctionObjectType& func) {
                if (IgnoreException_) {
                    try {
                        promise.SetValue(func());
                    } catch (...) {
                    }
                } else {
                    promise.SetValue(func());
                }
            }
        };

        template <bool IgnoreException_>
        struct TExecutor<void, IgnoreException_> {
            static void SetPromise(TPromise<void>& promise, const TFunctionObjectType& func) {
                if (IgnoreException_) {
                    try {
                        func();
                        promise.SetValue();
                    } catch (...) {
                    }
                } else {
                    func();
                    promise.SetValue();
                }
            }
        };

        void DoExecute() override {
            TExecutor<TResult, IgnoreException>::SetPromise(Result_, Func_);
        }
    };

}
