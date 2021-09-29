#include "bt_exception.h"
#include "yexception.h"

#include <util/system/backtrace.h>
#include <util/system/type_name.h>

#include <cxxabi.h>

#include <stdexcept>

#include <cstdio>

TString FormatExc(const std::exception& exception) {
    return TString::Join(TStringBuf("("), TypeName(exception), TStringBuf(") "), exception.what());
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

bool UncaughtException() noexcept {
// FIXME: use std::uncaught_exceptions() unconditionally after DEVTOOLS-8811
#if defined(__cpp_lib_uncaught_exceptions) && !defined(_LIBCPP_AVAILABILITY_UNCAUGHT_EXCEPTIONS)
    return std::uncaught_exceptions() > 0;
#else
    return std::uncaught_exception();
#endif
}

std::string CurrentExceptionTypeName() {
#if defined(_linux_) || defined(_darwin_)
    std::type_info* currentExceptionTypePtr = abi::__cxa_current_exception_type();
    if (currentExceptionTypePtr) {
        return TypeName(*currentExceptionTypePtr);
    }
#endif
    //There is no abi::__cxa_current_exception_type() on Windows.
    //Emulated it with rethrow - catch construction.
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

    exc << TStringBuf("(");
    exc << TStringBuf(LastSystemErrorText(Status_));
    exc << TStringBuf(") ");
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
