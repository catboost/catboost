#pragma once

typedef unsigned long (*TGetAuxVal)(unsigned long);

namespace NUbuntuCompat {
    class TGlibc {
    public:
        TGlibc() noexcept;
        ~TGlibc() noexcept;
        unsigned long GetAuxVal(unsigned long item) noexcept;
        bool IsSecure() noexcept;

    private:
        void* AuxVectorBegin;
        void* AuxVectorEnd;
        bool Secure;
    };

    TGlibc& GetGlibc() noexcept;
}
