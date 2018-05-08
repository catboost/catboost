#pragma once

#include "fwd.h"

#include <util/generic/ptr.h>

namespace NPar {
    // Local executor task registry, provideds an abstraction to create syncronization
    // barrier for tasks executed by local executor.
    //
    // Typical usage:
    // ```
    // TLocalExecutor e;
    // const auto r = MakeRegistry(...);
    //
    // e.Exec(r->Register([](int){ ... }, ...);
    // e.Exec(r->Register([](int){ ... }, ...);
    //
    // r->Wait();
    //
    // e.Exec(r->Register([](int){ ... }, ...);
    // e.Exec(r->Register([](int){ ... }, ...);
    //
    // r->Wait();
    // ```
    //
    class IWaitableRegistry : public TThrRefBase {
    public:
        virtual ~IWaitableRegistry() = default;

        // Register task in registry, thus making it waitable via `Wait` call.
        //
        // NOTE: returned object is not reusable, e.g. if you want to wait for a certain
        // function N times you will need to register it N times.
        //
        // NOTE: Returned object should either be executed or you should call `Reset` for registry.
        //
        virtual TIntrusivePtr<ILocallyExecutable> Register(TLocallyExecutableFunction) = 0;
        virtual TIntrusivePtr<ILocallyExecutable> Register(TIntrusivePtr<::NPar::ILocallyExecutable>) = 0;

        // Block caller until all tasks registred in current registry are complete.
        //
        virtual void Wait() = 0;

        // Reset registry state, all registred tasks will be unregistred from current registry.
        //
        virtual void Reset() = 0;
    };

    TIntrusivePtr<IWaitableRegistry> MakeDefaultWaitableRegistry();
}
