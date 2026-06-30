#ifndef LEAKY_REF_COUNTED_SINGLETON_INL_H_
#error "Direct inclusion of this file is not allowed, include leaky_ref_counted_singleton.h"
// For the sake of sane code completion.
#include "leaky_ref_counted_singleton.h"
#endif

#include "new.h"

#include <atomic>
#include <mutex>

#include <util/system/compiler.h>
#include <util/system/sanitizers.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, class... TArgs>
TIntrusivePtr<T> LeakyRefCountedSingleton(TArgs&&... args)
{
    static std::atomic<T*> Ptr;
    auto* ptr = Ptr.load(std::memory_order::acquire);
    if (Y_LIKELY(ptr)) {
        return ptr;
    }

    static std::once_flag Initialized;
    std::call_once(Initialized, [&] {
        auto ptr = New<T>(std::forward<TArgs>(args)...);
        Ref(ptr.Get());
        Ptr.store(ptr.Get());
    });

    return Ptr.load();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
