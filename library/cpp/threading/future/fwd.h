#pragma once

namespace NThreading {
    struct TFutureException;

    template <typename T>
    class TFuture;

    template <typename T>
    class TPromise;

    template <typename TR = void, bool IgnoreException = false>
    class TLegacyFuture;
}
