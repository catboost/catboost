#pragma once

#include <util/generic/yexception.h>
#include <util/generic/ptr.h>

namespace NDns {
    class IError {
    public:
        virtual ~IError() = default;

        virtual void Raise() = 0;
    };

    typedef TAutoPtr<IError> IErrorRef;

    IErrorRef SaveError();
}
