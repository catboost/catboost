#include "err.h"

#include <util/system/tls.h>

// This file has to be GLOBAL to initialize thrlocal before other global
// initializers call ERR_get_state.

static Y_POD_THREAD(ERR_STATE*) thrlocal(nullptr);

extern "C" ERR_STATE* y_openssl_err_get_thrlocal() {
    return thrlocal;
}

extern "C" void y_openssl_err_set_thrlocal(ERR_STATE* state) {
    thrlocal = state;
}
