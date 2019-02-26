#include "ecdsa.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

void ECDSA_SIG_get0(const ECDSA_SIG *sig, const BIGNUM **pr, const BIGNUM **ps)
{
    if (pr != NULL)
        *pr = sig->r;
    if (ps != NULL)
        *ps = sig->s;
}

int ECDSA_SIG_set0(ECDSA_SIG *sig, BIGNUM *r, BIGNUM *s)
{
    if (r == NULL || s == NULL)
        return 0;
    BN_clear_free(sig->r);
    BN_clear_free(sig->s);
    sig->r = r;
    sig->s = s;
    return 1;
}

#endif

#if OPENSSL_VERSION_NUMBER < 0x10101000L

const BIGNUM *ECDSA_SIG_get0_r(const ECDSA_SIG *sig)
{
    const BIGNUM *r;
    ECDSA_SIG_get0(sig, &r, NULL);
    return r;
}

const BIGNUM *ECDSA_SIG_get0_s(const ECDSA_SIG *sig)
{
    const BIGNUM *s;
    ECDSA_SIG_get0(sig, NULL, &s);
    return s;
}

#endif
