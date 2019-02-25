#include "bn.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

/* Bits of security, see SP800-57 */

int BN_security_bits(int L, int N)
{
    int secbits, bits;
    if (L >= 15360)
        secbits = 256;
    else if (L >= 7680)
        secbits = 192;
    else if (L >= 3072)
        secbits = 128;
    else if (L >= 2048)
        secbits = 112;
    else if (L >= 1024)
        secbits = 80;
    else
        return 0;
    if (N == -1)
        return secbits;
    bits = N / 2;
    if (bits < 80)
        return 0;
    return bits >= secbits ? secbits : bits;
}

#endif
