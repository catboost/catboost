#include <sys/auxv.h>

#include "glibc.h"
#include "features.h"

extern "C" {
    unsigned long getauxval(unsigned long item) noexcept {
        return NUbuntuCompat::GetGlibc().GetAuxVal(item);
    }
}
