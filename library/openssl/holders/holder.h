#pragma once

#include <util/generic/yexception.h>

namespace NOpenSSL {

template <typename TType, auto Create, auto Destroy, class... Args>
class THolder {
public:
    inline THolder(Args... args) {
        Ptr = Create(args...);
        if (!Ptr) {
            throw std::bad_alloc();
        }
    }

    THolder(const THolder&) = delete;
    THolder& operator=(const THolder&) = delete;

    inline ~THolder() noexcept {
        Destroy(Ptr);
    }

    inline operator TType* () noexcept {
        return Ptr;
    }

private:
    TType* Ptr;
};

}
