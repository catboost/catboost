#include "crypto.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

#include <string.h>

void *CRYPTO_zalloc(size_t num, const char *file, int line)
{
    void *ret = CRYPTO_malloc(num, file, line);

    if (ret != NULL)
        memset(ret, 0, num);
    return ret;
}

#endif
