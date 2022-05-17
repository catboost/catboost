#include "future.h"

namespace NThreading::NImpl {
    [[noreturn]] void ThrowFutureException(TStringBuf message, const TSourceLocation& source) {
        throw source + TFutureException() << message;
    }
}
