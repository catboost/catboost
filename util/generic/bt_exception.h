#pragma once

#include <utility>
#include "yexception.h"

#include <util/system/backtrace.h>

template <class T>
class TWithBackTrace: public T {
public:
    template <typename... Args>
    inline TWithBackTrace(Args&&... args)
        : T(std::forward<Args>(args)...)
    {
        BT_.Capture();
    }

    const TBackTrace* BackTrace() const noexcept override {
        return &BT_;
    }

private:
    TBackTrace BT_;
};
