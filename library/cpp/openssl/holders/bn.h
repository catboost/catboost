#pragma once

#include "holder.h"

#include <contrib/libs/openssl/include/openssl/bn.h>

namespace NOpenSSL {
    class TBignum : public THolder<BIGNUM, BN_new, BN_clear_free> {
    };

    class TBnCtx : public THolder<BN_CTX, BN_CTX_new, BN_CTX_free> {
    };
}
