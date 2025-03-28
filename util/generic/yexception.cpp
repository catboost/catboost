#include "yexception.h"

#include <util/system/backtrace.h>
#include <util/system/type_name.h>

#if defined(_linux_) || defined(_android_) || defined(_darwin_)
    #include <cxxabi.h>
#endif

#include <stdexcept>

#include <cstdio>

static void FormatExceptionTo(IOutputStream& out, const std::exception& exception) {
    out << "(" << TypeName(exception) << ") " << exception.what();
}

TString FormatExc(const std::exception& exception) {
    TStringStream out;
    FormatExceptionTo(out, exception);
    return out.Str();
}

TString CurrentExceptionMessage() {
    auto exceptionPtr = std::current_exception();
    if (exceptionPtr) {
        try {
            std::rethrow_exception(exceptionPtr);
        } catch (const yexception& e) {
            const TBackTrace* bt = e.BackTrace();

            if (bt) {
                return TString::Join(bt->PrintToString(), TStringBuf("\n"), FormatExc(e));
            }

            return FormatExc(e);
        } catch (const std::exception& e) {
            return FormatExc(e);
        } catch (...) {
        }

        return "unknown error";
    }

    return "(NO EXCEPTION)";
}

Y_DECLARE_UNUSED static void FormatBackTraceTo(IOutputStream& out, const TBackTrace& backtrace) {
    if (backtrace.size() == 0) {
        out << "backtrace is empty";
        return;
    }
    try {
        backtrace.PrintTo(out);
    } catch (const std::exception& e) {
        out << "Failed to print backtrace: ";
        FormatExceptionTo(out, e);
    }
}

void FormatCurrentExceptionTo(IOutputStream& out) {
    auto exceptionPtr = std::current_exception();

    if (Y_UNLIKELY(!exceptionPtr)) {
        out << "(NO EXCEPTION)\n";
        return;
    }

#ifdef _YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE
    TBackTrace backtrace = TBackTrace::FromCurrentException();
#endif
    try {
        std::rethrow_exception(exceptionPtr);
    } catch (const std::exception& e) {
        out << "Caught:\n";
        FormatExceptionTo(out, e);
        out << "\n";
#ifdef _YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE
        FormatBackTraceTo(out, backtrace);
#endif
        return;
    } catch (...) {
        out << "unknown error\n";
        return;
    }
}

TString FormatCurrentException() {
    TStringStream out;
    FormatCurrentExceptionTo(out);
    return out.Str();
}

bool UncaughtException() noexcept {
    return std::uncaught_exceptions() > 0;
}

std::string CurrentExceptionTypeName() {
#if defined(_linux_) || defined(_android_) || defined(_darwin_)
    std::type_info* currentExceptionTypePtr = abi::__cxa_current_exception_type();
    if (currentExceptionTypePtr) {
        return TypeName(*currentExceptionTypePtr);
    }
#endif
    // There is no abi::__cxa_current_exception_type() on Windows.
    // Emulated it with rethrow - catch construction.
    std::exception_ptr currentException = std::current_exception();
    Y_ASSERT(currentException != nullptr);
    try {
        std::rethrow_exception(currentException);
    } catch (const std::exception& e) {
        return TypeName(typeid(e));
    } catch (...) {
        return "unknown type";
    }
}

void TSystemError::Init() {
    yexception& exc = *this;

    exc << "(Error "sv
        << Status_
        << ": "sv
        << TStringBuf(LastSystemErrorText(Status_))
        << ") "sv;
}

NPrivateException::yexception::yexception() {
    ZeroTerminate();
}

TStringBuf NPrivateException::yexception::AsStrBuf() const {
    if (Buf_.Left()) {
        return TStringBuf(Buf_.Data(), Buf_.Filled());
    }

    return TStringBuf(Buf_.Data(), Buf_.Filled() - 1);
}

void NPrivateException::yexception::ZeroTerminate() noexcept {
    char* end = (char*)Buf_.Current();

    if (!Buf_.Left()) {
        --end;
    }

    *end = 0;
}

const char* NPrivateException::yexception::what() const noexcept {
    return Buf_.Data();
}

const TBackTrace* NPrivateException::yexception::BackTrace() const noexcept {
    return nullptr;
}

void fputs(const std::exception& e, FILE* f) {
    char message[256];
    size_t len = Min(sizeof(message) - 2, strlcpy(message, e.what(), sizeof(message) - 1));
    message[len++] = '\n';
    message[len] = 0;
    fputs(message, f);
}

void ::NPrivate::ThrowYException(const ::NPrivate::TSimpleExceptionMessage& sm) {
    throw sm.Location + yexception() << sm.Message;
}

void ::NPrivate::ThrowYExceptionWithBacktrace(const ::NPrivate::TSimpleExceptionMessage& sm) {
    throw sm.Location + TWithBackTrace<yexception>() << sm.Message;
}
