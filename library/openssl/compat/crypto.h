#pragma once

#include <contrib/libs/openssl/include/openssl/crypto.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

# define OPENSSL_zalloc(num)     CRYPTO_zalloc((int)num,__FILE__,__LINE__)

void *CRYPTO_zalloc(size_t num, const char *file, int line);

#endif

#ifdef  __cplusplus
}
#endif
