#pragma once

#include <contrib/libs/openssl/include/openssl/evp.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

int EVP_PKEY_up_ref(EVP_PKEY *pkey);

#endif

#ifdef  __cplusplus
}
#endif
