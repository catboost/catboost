#pragma once

#include <stlfwd>

namespace NPar {
    struct ILocallyExecutable;

    // Alternative and more simple way of describing a job for executor. Function argument has the
    // same meaning as `id` in `ILocallyExecutable::LocalExec`.
    //
    using TLocallyExecutableFunction = std::function<void(int)>;

    class TLocalExecutor;

    class IWaitableRegistry;
}
