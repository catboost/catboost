#pragma once

#include "strbuf.h"
#include "string.h"
#include "utility.h"
#include "va_args.h"
#include <utility>

#include <util/system/defaults.h>
#include <util/system/error.h>
#include <util/system/compat.h>
#include <util/system/src_location.h>
#include <util/stream/tempbuf.h>

#include <exception>

#include <cstdio>

class TBackTrace;

namespace NPrivateException {
    class yexception: public std::exception {
    public:
        const char* what() const noexcept override;
        virtual const TBackTrace* BackTrace() const noexcept;

        template <class T>
        inline void Append(const T& t) {
            TTempBufWrapperOutput tempBuf(Buf_);
            static_cast<IOutputStream&>(tempBuf) << t;
        }

        TStringBuf AsStrBuf() const {
            return TStringBuf(Buf_.Data(), Buf_.Filled());
        }

    private:
        mutable TTempBuf Buf_;
    };

    template <class E, class T>
    static inline E&& operator<<(E&& e, const T& t) {
        e.Append(t);

        return std::forward<E>(e);
    }

    template <class T>
    static inline T&& operator+(const TSourceLocation& sl, T&& t) {
        return std::forward<T>(t << sl << STRINGBUF(": "));
    }
}

class yexception: public NPrivateException::yexception {
};

Y_DECLARE_OUT_SPEC(inline, yexception, stream, value) {
    stream << value.AsStrBuf();
}

namespace NPrivate {
    class TSystemErrorStatus {
    public:
        inline TSystemErrorStatus()
            : Status_(LastSystemError())
        {
        }

        inline TSystemErrorStatus(int status)
            : Status_(status)
        {
        }

        inline int Status() const noexcept {
            return Status_;
        }

    private:
        int Status_;
    };
}

class TSystemError: public virtual NPrivate::TSystemErrorStatus, public virtual yexception {
public:
    inline TSystemError(int status)
        : TSystemErrorStatus(status)
    {
        Init();
    }

    inline TSystemError() {
        Init();
    }

private:
    void Init();
};

struct TIoException: public virtual yexception {
};

class TIoSystemError: public TSystemError, public TIoException {
};

class TFileError: public TIoSystemError {
};

struct TBadCastException: public virtual yexception {
};

#define ythrow throw __LOCATION__ +

void fputs(const std::exception& e, FILE* f = stderr);

TString CurrentExceptionMessage();
bool UncaughtException() noexcept;

void ThrowBadAlloc();
void ThrowLengthError(const char* descr);
void ThrowRangeError(const char* descr);

#define Y_ENSURE_EX(CONDITION, THROW_EXPRESSION) \
    do {                                         \
        if (Y_UNLIKELY(!(CONDITION))) {          \
            ythrow THROW_EXPRESSION;             \
        }                                        \
    } while (false)

#define Y_ENSURE_IMPL_1(CONDITION) Y_ENSURE_EX(CONDITION, yexception() << STRINGBUF("Condition violated: `" Y_STRINGIZE(CONDITION) "'"))
#define Y_ENSURE_IMPL_2(CONDITION, MESSAGE) Y_ENSURE_EX(CONDITION, yexception() << MESSAGE)

/**
 * @def Y_ENSURE
 *
 * This macro is inteded to use as a shortcut for `if () { throw }`.
 *
 * @code
 * void DoSomethingLovely(const int x, const int y) {
 *     Y_ENSURE(x > y, "`x` must be greater than `y`");
 *     Y_ENSURE(x > y); // if you are too lazy
 *     // actually doing something nice here
 * }
 * @endcode
 */
#define Y_ENSURE(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, Y_ENSURE_IMPL_2, Y_ENSURE_IMPL_1)(__VA_ARGS__))
