#include "magic.h"

#include <util/generic/yexception.h>

using namespace NDns;

namespace {
    namespace NX {
        struct TError: public IError {
            inline TError()
                : E_(std::current_exception())
            {
            }

            void Raise() override {
                std::rethrow_exception(E_);
            }

            std::exception_ptr E_;
        };
    }
}

IErrorRef NDns::SaveError() {
    using namespace NX;

    return new NX::TError();
}
