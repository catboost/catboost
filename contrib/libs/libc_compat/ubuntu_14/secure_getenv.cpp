#include <stdlib.h>

#include "glibc.h"

extern "C" {
    char *secure_getenv(const char *name) noexcept {
            if (NUbuntuCompat::GetGlibc().IsSecure()) {
                return nullptr;
            }
            return getenv(name);
    }
}
