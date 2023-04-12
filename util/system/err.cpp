#include "defaults.h"
#include "progname.h"
#include "compat.h"
#include "error.h"

#include <util/generic/scope.h>

#include <util/system/compat.h>
#include <util/stream/printf.h>
#include <util/stream/output.h>

void vwarnx(const char* fmt, va_list args) {
    Cerr << GetProgramName() << ": ";

    if (fmt) {
        Printf(Cerr, fmt, args);
    }

    Cerr << '\n';
}

void vwarn(const char* fmt, va_list args) {
    int curErrNo = errno;
    auto curErrText = LastSystemErrorText();

    Y_DEFER {
        errno = curErrNo;
    };

    Cerr << GetProgramName() << ": ";

    if (fmt) {
        Printf(Cerr, fmt, args);
        Cerr << ": ";
    }

    Cerr << curErrText << '\n';
}

void warn(const char* fmt, ...) {
    va_list args;

    va_start(args, fmt);
    vwarn(fmt, args);
    va_end(args);
}

void warnx(const char* fmt, ...) {
    va_list args;

    va_start(args, fmt);
    vwarnx(fmt, args);
    va_end(args);
}

[[noreturn]] void verr(int status, const char* fmt, va_list args) {
    vwarn(fmt, args);
    std::exit(status);
}

[[noreturn]] void err(int status, const char* fmt, ...) {
    va_list args;

    va_start(args, fmt);
    verr(status, fmt, args);
    va_end(args);
}

[[noreturn]] void verrx(int status, const char* fmt, va_list args) {
    vwarnx(fmt, args);
    std::exit(status);
}

[[noreturn]] void errx(int status, const char* fmt, ...) {
    va_list args;

    va_start(args, fmt);
    verrx(status, fmt, args);
    va_end(args);
}
