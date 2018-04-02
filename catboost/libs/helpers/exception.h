#pragma once

#include <util/generic/yexception.h>

class TCatboostException : public yexception {

};

#define CB_ENSURE_IMPL_1(CONDITION) Y_ENSURE_EX(CONDITION, TCatboostException() << AsStringBuf("Condition violated: `" Y_STRINGIZE(CONDITION) "'"))
#define CB_ENSURE_IMPL_2(CONDITION, MESSAGE) Y_ENSURE_EX(CONDITION, TCatboostException() << MESSAGE)

#define CB_ENSURE(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, CB_ENSURE_IMPL_2, CB_ENSURE_IMPL_1)(__VA_ARGS__))

// Kudos to YT team :)
template <class TCallback>
class TFinallyGuard
{
public:
    template <class T>
    explicit TFinallyGuard(T&& finally)
        : Finally_(std::forward<T>(finally))
    { }

    TFinallyGuard(TFinallyGuard&& guard)
        : Released_(guard.Released_)
        , Finally_(std::move(guard.Finally_))
    {
        guard.Release();
    }

    TFinallyGuard(const TFinallyGuard&) = delete;
    TFinallyGuard& operator=(const TFinallyGuard&) = delete;
    TFinallyGuard& operator=(TFinallyGuard&&) = delete;

    void Release()
    {
        Released_ = true;
    }

    ~TFinallyGuard()
    {
        if (!Released_) {
            Finally_();
        }
    }

private:
    bool Released_ = false;
    TCallback Finally_;
};

template <class TCallback>
TFinallyGuard<std::decay_t<TCallback>> Finally(TCallback&& callback)
{
    return TFinallyGuard<std::decay_t<TCallback>>(std::forward<TCallback>(callback));
}
