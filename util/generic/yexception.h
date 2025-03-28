#pragma once

#include "strbuf.h"
#include "string.h"
#include "utility.h"
#include "va_args.h"

#include <util/stream/tempbuf.h>
#include <util/system/backtrace.h>
#include <util/system/compat.h>
#include <util/system/compiler.h>
#include <util/system/defaults.h>
#include <util/system/error.h>
#include <util/system/src_location.h>
#include <util/system/platform.h>

#include <exception>
#include <utility>
#include <cstdio>

namespace NPrivateException {
    class TTempBufCuttingWrapperOutput: public IOutputStream {
    public:
        TTempBufCuttingWrapperOutput(TTempBuf& tempbuf)
            : TempBuf_(tempbuf)
        {
        }

        void DoWrite(const void* data, size_t len) override {
            TempBuf_.Append(data, Min(len, TempBuf_.Left()));
        }

    private:
        TTempBuf& TempBuf_;
    };

    class yexception: public std::exception {
    public:
        yexception();
        yexception(const yexception&) = default;
        yexception(yexception&&) = default;

        yexception& operator=(const yexception&) = default;
        yexception& operator=(yexception&&) = default;

        const char* what() const noexcept override;
        virtual const TBackTrace* BackTrace() const noexcept;

        template <class T>
        inline void Append(const T& t) {
            TTempBufCuttingWrapperOutput tempBuf(Buf_);
            static_cast<IOutputStream&>(tempBuf) << t;
            ZeroTerminate();
        }

        TStringBuf AsStrBuf() const Y_LIFETIME_BOUND;

    private:
        void ZeroTerminate() noexcept;

    private:
        TTempBuf Buf_;
    };

    template <class E, class T>
    static inline std::enable_if_t<std::is_base_of<yexception, std::decay_t<E>>::value, E&&>
    operator<<(E&& e Y_LIFETIME_BOUND, const T& t) {
        e.Append(t);

        return std::forward<E>(e);
    }

    template <class T>
    static inline T&& operator+(const TSourceLocation& sl, T&& t Y_LIFETIME_BOUND) {
        return std::forward<T>(t << sl << TStringBuf(": "));
    }
} // namespace NPrivateException

class yexception: public NPrivateException::yexception {
};

Y_DECLARE_OUT_SPEC(inline, yexception, stream, value) {
    stream << value.AsStrBuf();
}

class TSystemError: public yexception {
public:
    TSystemError(int status)
        : Status_(status)
    {
        Init();
    }

    TSystemError()
        : TSystemError(LastSystemError())
    {
    }

    int Status() const noexcept {
        return Status_;
    }

private:
    void Init();

private:
    int Status_;
};

class TIoException: public TSystemError {
};

class TIoSystemError: public TIoException {
};

class TFileError: public TIoSystemError {
};

/**
 * TBadArgumentException should be thrown when an argument supplied to some function (or constructor)
 * is invalid or incorrect.
 *
 * \note
 * A special case when such argument is given to a function which performs type casting
 * (e.g. integer from string) is covered by the TBadCastException class which is derived from
 * TBadArgumentException.
 */
struct TBadArgumentException: public virtual yexception {
};

/**
 * TBadCastException should be thrown to indicate the failure of some type casting procedure
 * (e.g. reading an integer parameter from string).
 */
struct TBadCastException: public virtual TBadArgumentException {
};

template <class T>
class TWithBackTrace: public T {
public:
    template <typename... Args>
    inline TWithBackTrace(Args&&... args)
        : T(std::forward<Args>(args)...)
    {
        BT_.Capture();
    }

    const TBackTrace* BackTrace() const noexcept override {
        return &BT_;
    }

private:
    TBackTrace BT_;
};

#define ythrow throw __LOCATION__ +

namespace NPrivate {
    /// Encapsulates data for one of the most common case in which
    /// exception message consists of single constant string
    struct TSimpleExceptionMessage {
        TSourceLocation Location;
        TStringBuf Message;
    };

    [[noreturn]] void ThrowYException(const TSimpleExceptionMessage& sm);
    [[noreturn]] void ThrowYExceptionWithBacktrace(const TSimpleExceptionMessage& sm);
} // namespace NPrivate

void fputs(const std::exception& e, FILE* f = stderr);

TString CurrentExceptionMessage();

/**
 * Formats current exception for logging purposes. Includes formatted backtrace if it is stored
 * alongside the exception.
 * The output format is a subject to change, do not depend or canonize it.
 * The speed of this method is not guaranteed either. Do not call it in hot paths of your code.
 *
 * The lack of current exception prior to the invocation indicates logical bug in the client code.
 * Y_ABORT_UNLESS asserts the existence of exception, otherwise panic and abort.
 */
TString FormatCurrentException();
void FormatCurrentExceptionTo(IOutputStream& out);

/*
 * A neat method that detects whether stack unwinding is in progress.
 * As its std counterpart (that is std::uncaught_exception())
 * was removed from the standard, this method uses std::uncaught_exceptions() internally.
 *
 * If you are struggling to use this method, please, consider reading
 *
 * http://www.gotw.ca/gotw/047.htm
 * and
 * http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4152.pdf
 *
 * DO NOT USE THIS METHOD IN DESTRUCTORS.
 */
bool UncaughtException() noexcept;

std::string CurrentExceptionTypeName();

TString FormatExc(const std::exception& exception);

#define Y_THROW_UNLESS_EX(CONDITION, THROW_EXPRESSION) \
    do {                                               \
        if (Y_UNLIKELY(!(CONDITION))) {                \
            ythrow THROW_EXPRESSION;                   \
        }                                              \
    } while (false)
#define Y_ENSURE_EX Y_THROW_UNLESS_EX

/// @def Y_ENSURE_SIMPLE
/// This macro works like the Y_ENSURE, but requires the second argument to be a constant string view.
/// Should not be used directly.
#define Y_ENSURE_SIMPLE(CONDITION, MESSAGE, THROW_FUNCTION)                                                                 \
    do {                                                                                                                    \
        if (Y_UNLIKELY(!(CONDITION))) {                                                                                     \
            /* use variable to guarantee evaluation at compile time */                                                      \
            static constexpr const ::NPrivate::TSimpleExceptionMessage __SIMPLE_EXCEPTION_MESSAGE{__LOCATION__, (MESSAGE)}; \
            THROW_FUNCTION(__SIMPLE_EXCEPTION_MESSAGE);                                                                     \
        }                                                                                                                   \
    } while (false)

#define Y_ENSURE_IMPL_1(CONDITION) Y_ENSURE_SIMPLE(CONDITION, ::TStringBuf("Condition violated: `" Y_STRINGIZE(CONDITION) "'"), ::NPrivate::ThrowYException)
#define Y_ENSURE_IMPL_2(CONDITION, MESSAGE) Y_ENSURE_EX(CONDITION, yexception() << MESSAGE)

#define Y_ENSURE_BT_IMPL_1(CONDITION) Y_ENSURE_SIMPLE(CONDITION, ::TStringBuf("Condition violated: `" Y_STRINGIZE(CONDITION) "'"), ::NPrivate::ThrowYExceptionWithBacktrace)
#define Y_ENSURE_BT_IMPL_2(CONDITION, MESSAGE) Y_ENSURE_EX(CONDITION, TWithBackTrace<yexception>() << MESSAGE)

/**
 * @def Y_ENSURE
 *
 * This macro is intended to be used as a shortcut for `if () { throw }`.
 *
 * @code
 * void DoSomethingLovely(const int x, const int y) {
 *     Y_ENSURE(x > y, "`x` must be greater than `y`");
 *     Y_ENSURE(x > y); // if you are too lazy
 *     // actually doing something nice here
 * }
 * @endcode
 */
#define Y_THROW_UNLESS(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, Y_ENSURE_IMPL_2, Y_ENSURE_IMPL_1)(__VA_ARGS__))
#define Y_ENSURE Y_THROW_UNLESS

/**
 * @def Y_ENSURE_BT
 *
 * This macro is intended to be used as a shortcut for `if () { throw TWithBackTrace<yexception>() << "message"; }`.
 *
 * @code
 * void DoSomethingLovely(const int x, const int y) {
 *     Y_ENSURE_BT(x > y, "`x` must be greater than `y`");
 *     Y_ENSURE_BT(x > y); // if you are too lazy
 *     // actually doing something nice here
 * }
 * @endcode
 */
#define Y_ENSURE_BT(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, Y_ENSURE_BT_IMPL_2, Y_ENSURE_BT_IMPL_1)(__VA_ARGS__))
