#include <stdlib.h>
#include <exception>
#include <typeinfo>

#include <util/stream/output.h>
#include <util/system/backtrace.h>
#include <util/system/tls.h>
#include <util/generic/yexception.h>

#include "terminate_handler.h"

Y_POD_STATIC_THREAD(bool)
Terminating(false);

void FancyTerminateHandler() {
    if (Terminating) {
        Cerr << "FancyTerminateHandler called recursively\n";
        abort();
    }
    Terminating = true;

    if (std::current_exception()) {
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
