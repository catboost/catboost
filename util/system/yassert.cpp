#include "yassert.h"

#include "backtrace.h"
#include "guard.h"
#include "spinlock.h"
#include "src_root.h"

#include <util/generic/singleton.h>
#include <util/generic/string.h>
#include <util/stream/output.h>
#include <util/stream/str.h>
#include <util/string/printf.h>

#include <cstdlib>
#include <stdarg.h>
#include <stdio.h>

namespace {
    struct TPanicLockHolder: public TAdaptiveLock {
    };
}

void ::NPrivate::Panic(const TStaticBuf& file, int line, const char* function, const char* expr, const char* format, ...) noexcept {
    try {
        // We care of panic of first failed thread only
        // Otherwise stderr could contain multiple messages and stack traces shuffled
        auto guard = Guard(*Singleton<TPanicLockHolder>());

        TString errorMsg;
        va_list args;
        va_start(args, format);
        // format has " " prefix to mute GCC warning on empty format
        vsprintf(errorMsg, format[0] == ' ' ? format + 1 : format, args);
        va_end(args);

        TString r;
        TStringOutput o(r);
        if (expr) {
            o << "VERIFY failed: " << errorMsg << Endl;
        } else {
            o << "FAIL: " << errorMsg << Endl;
        }
        o << "  " << file.As<TStringBuf>() << ":" << line << Endl;
        if (expr) {
            o << "  " << function << "(): requirement " << expr << " failed" << Endl;
        } else {
            o << "  " << function << "() failed" << Endl;
        }
        Cerr << r;
#ifndef WITH_VALGRIND
        PrintBackTrace();
#endif
    } catch (...) {
        //nothing we can do here
    }

    abort();
}
