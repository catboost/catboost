#pragma once

#include <contrib/libs/openssl/include/openssl/evp.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

inline EVP_MD_CTX *EVP_MD_CTX_new(void) {
    return EVP_MD_CTX_create();
}

inline void EVP_MD_CTX_free(EVP_MD_CTX *ctx) {
    EVP_MD_CTX_destroy(ctx);
}

int EVP_PKEY_up_ref(EVP_PKEY *pkey);

#endif

#ifdef  __cplusplus
}
#endif
