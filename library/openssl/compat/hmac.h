#pragma once

#include <contrib/libs/openssl/include/openssl/hmac.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

HMAC_CTX *HMAC_CTX_new(void);
void HMAC_CTX_free(HMAC_CTX *ctx);

#endif

#ifdef  __cplusplus
}
#endif
