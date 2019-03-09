#include "hmac.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

#include "crypto.h"

HMAC_CTX *HMAC_CTX_new(void)
{
    HMAC_CTX *ctx = OPENSSL_zalloc(sizeof(HMAC_CTX));

    if (ctx != NULL) {
        HMAC_CTX_init(ctx);
    }
    return ctx;
}

void HMAC_CTX_free(HMAC_CTX* ctx) {
    if (ctx != NULL) {
        HMAC_CTX_cleanup(ctx);
        OPENSSL_free(ctx);
    }
}

#endif
