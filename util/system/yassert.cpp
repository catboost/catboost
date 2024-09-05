#include "yassert.h"

#include "backtrace.h"
#include "guard.h"
#include "spinlock.h"
#include "src_root.h"

#include <util/datetime/base.h>
#include <util/generic/singleton.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/stream/output.h>
#include <util/stream/str.h>
#include <util/string/printf.h>

#include <cstdlib>
#include <stdarg.h>
#include <stdio.h>

#ifdef CLANG_COVERAGE
extern "C" {
    // __llvm_profile_write_file may not be provided if the executable target uses NO_CLANG_COVERAGE() macro and
    // arrives as test's dependency via DEPENDS() macro.
    // That's why we provide a weak no-op implementation for __llvm_profile_write_file,
    // which is used below in the code, to correctly save codecoverage profile before program exits using abort().
    Y_WEAK int __llvm_profile_write_file(void) {
        return 0;
    }
}

#endif

namespace {
    struct TPanicLockHolder: public TAdaptiveLock {
    };
} // namespace
namespace NPrivate {
    [[noreturn]] Y_NO_INLINE void InternalPanicImpl(int line, const char* function, const char* expr, int, int, int, const TStringBuf file, const char* errorMessage, size_t errorMessageSize) noexcept;
} // namespace NPrivate

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

        constexpr int abiPlaceholder = 0;
        ::NPrivate::InternalPanicImpl(line, function, expr, abiPlaceholder, abiPlaceholder, abiPlaceholder, file.As<TStringBuf>(), errorMsg.c_str(), errorMsg.size());
    } catch (...) {
        // ¯\_(ツ)_/¯
    }

    abort();
}

namespace NPrivate {
    [[noreturn]] Y_NO_INLINE void InternalPanicImpl(int line, const char* function, const char* expr, int, int, int, const TStringBuf file, const char* errorMessage, size_t errorMessageSize) noexcept try {
        TStringBuf errorMsg{errorMessage, errorMessageSize};
        const TString now = TInstant::Now().ToStringLocal();

        TString r;
        TStringOutput o(r);
        if (expr) {
            o << "VERIFY failed (" << now << "): " << errorMsg << Endl;
        } else {
            o << "FAIL (" << now << "): " << errorMsg << Endl;
        }
        o << "  " << file << ":" << line << Endl;
        if (expr) {
            o << "  " << function << "(): requirement " << expr << " failed" << Endl;
        } else {
            o << "  " << function << "() failed" << Endl;
        }
        Cerr << r << Flush;
#ifndef WITH_VALGRIND
        PrintBackTrace();
#endif
#ifdef CLANG_COVERAGE
        if (__llvm_profile_write_file()) {
            Cerr << "Failed to dump clang coverage" << Endl;
        }
#endif
        abort();
    } catch (...) {
        abort();
    }
} // namespace NPrivate
