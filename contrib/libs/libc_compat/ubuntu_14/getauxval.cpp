#include <sys/auxv.h>

#include "glibc.h"
#include "features.h"

extern "C" {
    unsigned long __getauxval(unsigned long item) noexcept {
        return NUbuntuCompat::GetGlibc().GetAuxVal(item);
    }

    weak_alias(__getauxval, getauxval);
}
