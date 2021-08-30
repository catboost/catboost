#include "stack_guards.h"


namespace NCoro::NStack {

    template<>
    const TCanaryGuard& GetGuard<TCanaryGuard>() noexcept {
        static const TCanaryGuard guard;
        return guard;
    }

    template<>
    const TPageGuard& GetGuard<TPageGuard>() noexcept {
        static const TPageGuard guard;
        return guard;
    }
}