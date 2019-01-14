#include "init_atfork.h"
#include "random.h"
#include "entropy.h"

#include <util/generic/singleton.h>

#if defined(_unix_)
#include <pthread.h>
#endif

namespace {
    struct TInit {
        inline TInit() noexcept {
            (void)&AtFork;

#if defined(_unix_)
            Y_VERIFY(pthread_atfork(nullptr, AtFork, nullptr) == 0, "it happens");
#endif
        }

        static void AtFork() noexcept {
            ResetRandomState();
        }
    };
}

void RNGInitAtForkHandlers() {
    SingletonWithPriority<TInit, 0>();
}
