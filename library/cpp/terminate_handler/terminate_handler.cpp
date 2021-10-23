#include <cstdlib>
#include <exception>

#include <util/stream/output.h>
#include <util/system/backtrace.h>
#include <util/generic/yexception.h>

namespace {
    // Avoid infinite recursion if std::terminate is triggered anew by the
    // FancyTerminateHandler.
    thread_local int TerminateCount = 0;

    void FancyTerminateHandler() {
        switch (++TerminateCount) {
            case 1:
                break;
            case 2:
                Cerr << "FancyTerminateHandler called recursively" << Endl;
                [[fallthrough]];
            default:
                abort();
                break;
        }

        if (std::current_exception()) {
            Cerr << "Uncaught exception: " << CurrentExceptionMessage() << '\n';
        } else {
            Cerr << "Terminate for unknown reason (no current exception)\n";
        }
        PrintBackTrace();
        Cerr.Flush();
        abort();
    }

    [[maybe_unused]] auto _ = std::set_terminate(&FancyTerminateHandler);
}
