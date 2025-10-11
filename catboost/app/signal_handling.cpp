#include <catboost/private/libs/init/init_reg.h>
#include <catboost/libs/helpers/interrupt.h>

#include <util/system/compiler.h>
#include <util/system/interrupt_signals.h>
#include <util/system/yassert.h>

#include <atomic>
#include <csignal>


namespace NCB {

    static std::atomic<bool> HasBeenInterrupted = false;

    static void AppInterruptHandler() {
        if (HasBeenInterrupted) {
            throw TInterruptException();
        }
    }

    static void AppInterruptSignalHandler(int signum) {
        Y_ASSERT((signum == SIGINT) || (signum == SIGTERM) || (signum == SIGHUP));
        HasBeenInterrupted = true;
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
