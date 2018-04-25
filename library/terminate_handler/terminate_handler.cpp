#include <stdlib.h>
#include <exception>
#include <typeinfo>

#include <util/stream/output.h>
#include <util/system/backtrace.h>
#include <util/system/tls.h>
#include <util/generic/yexception.h>

#include "terminate_handler.h"

#ifdef __GNUC__
#include <cxxabi.h>
extern "C" std::type_info* __cxa_current_exception_type();
#endif

static bool UncaughtExceptionInTerminateHanlder() {
#ifdef __GNUC__
    // UncaughtException() returns false on Linux
    return !!__cxa_current_exception_type();
#else
    return UncaughtException();
#endif
}

Y_POD_STATIC_THREAD(bool)
Terminating(false);

void FancyTerminateHandler() {
    if (Terminating) {
        Cerr << "FancyTerminateHandler called recursively\n";
        abort();
    }
    Terminating = true;

    if (UncaughtExceptionInTerminateHanlder()) {
        Cerr << "Uncaught exception: " << CurrentExceptionMessage() << Endl;
    } else {
        Cerr << "Terminate for unknown reason\n";
    }
    PrintBackTrace();
    abort();
}

void SetFancyTerminateHandler() {
    std::set_terminate(&FancyTerminateHandler);
}
