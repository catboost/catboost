#include "evp.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

int EVP_PKEY_up_ref(EVP_PKEY *pkey)
{
    CRYPTO_add(&pkey->references, 1, CRYPTO_LOCK_EVP_PKEY);
    return 1;
}

#endif
