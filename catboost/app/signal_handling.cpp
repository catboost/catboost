#include <catboost/private/libs/init/init_reg.h>
#include <catboost/libs/helpers/interrupt.h>

#include <util/system/atomic.h>
#include <util/system/compiler.h>
#include <util/system/interrupt_signals.h>
#include <util/system/yassert.h>

#include <csignal>


namespace NCB {

    static volatile TAtomic HasBeenInterrupted = 0;

    static void AppInterruptHandler() {
        if (AtomicGet(HasBeenInterrupted)) {
            throw TInterruptException();
        }
    }

    static void AppInterruptSignalHandler(int signum) {
        Y_ASSERT((signum == SIGINT) || (signum == SIGTERM) || (signum == SIGHUP));
        AtomicSet(HasBeenInterrupted, 1);
    }

    static NCB::TCmdLineInit::TRegistrator SetupAppSignalHandling(
        [](int argc, const char* argv[]) {
            Y_UNUSED(argc);
            Y_UNUSED(argv);

            SetInterruptHandler(AppInterruptHandler);
            SetInterruptSignalsHandler(AppInterruptSignalHandler);
        }
    );

}
