#pragma once

#include "holder.h"

#include <library/openssl/compat/hmac.h> // Remove after OpenSSL upgrade

#include <contrib/libs/openssl/include/openssl/hmac.h>

namespace NOpenSSL {
    class THmacCtx : public THolder<HMAC_CTX, HMAC_CTX_new, HMAC_CTX_free> {
    };
}
