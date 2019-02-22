#pragma once

#include "holder.h"

#include <contrib/libs/openssl/include/openssl/hmac.h>

namespace {
    HMAC_CTX* HMAC_CTX_new() {
        HMAC_CTX* ctx = static_cast<HMAC_CTX*>(OPENSSL_malloc(sizeof(HMAC_CTX)));
        if (ctx) {
            HMAC_CTX_init(ctx);
        }
        return ctx;
    }

    void HMAC_CTX_free(HMAC_CTX* ctx) {
        HMAC_CTX_cleanup(ctx);
        OPENSSL_free(ctx);
    }
}

namespace NOpenSSL {
    class THmacCtx : public THolder<HMAC_CTX, HMAC_CTX_new, HMAC_CTX_free> {
    };
}
