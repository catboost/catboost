#pragma once

#include "fwd.h"

#include <library/cpp/threading/future/core/future.h>
#include <library/cpp/threading/future/wait/wait_group.h>

namespace NThreading {
    ////////////////////////////////////////////////////////////////////////////////

    // waits for all futures
    [[nodiscard]] TFuture<void> WaitAll(const TFuture<void>& f1);
    [[nodiscard]] TFuture<void> WaitAll(const TFuture<void>& f1, const TFuture<void>& f2);
    template <typename TContainer>
    [[nodiscard]] TFuture<void> WaitAll(const TContainer& futures);

    // waits for the first exception or for all futures
    [[nodiscard]] TFuture<void> WaitExceptionOrAll(const TFuture<void>& f1);
    [[nodiscard]] TFuture<void> WaitExceptionOrAll(const TFuture<void>& f1, const TFuture<void>& f2);
    template <typename TContainer>
    [[nodiscard]] TFuture<void> WaitExceptionOrAll(const TContainer& futures);

    // waits for any future
    [[nodiscard]] TFuture<void> WaitAny(const TFuture<void>& f1);
    [[nodiscard]] TFuture<void> WaitAny(const TFuture<void>& f1, const TFuture<void>& f2);
    template <typename TContainer>
    [[nodiscard]] TFuture<void> WaitAny(const TContainer& futures);
}

#define INCLUDE_FUTURE_INL_H
#include "wait-inl.h"
#undef INCLUDE_FUTURE_INL_H
