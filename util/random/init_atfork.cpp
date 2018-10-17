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
            (void)AtFork;

#if defined(_unix_)
            Y_ENSURE(pthread_atfork(0, 0, AtFork) == 0, "it happens");
#endif
        }

        static void AtFork() noexcept {
            ResetEntropyPool();
            ResetRandomState();
        }
    };
}

void RNGInitAtForkHandlers() {
    SingletonWithPriority<TInit, 0>();
}
