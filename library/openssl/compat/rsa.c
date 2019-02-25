#include "rsa.h"

#include "bn.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

int RSA_bits(const RSA *r)
{
    return BN_num_bits(r->n);
}

int RSA_security_bits(const RSA *rsa)
{
    return BN_security_bits(BN_num_bits(rsa->n), -1);
}

int RSA_set0_key(RSA *r, BIGNUM *n, BIGNUM *e, BIGNUM *d)
{
    /* If the fields n and e in r are NULL, the corresponding input
     * parameters MUST be non-NULL for n and e.  d may be
     * left NULL (in case only the public key is used).
     */
    if ((r->n == NULL && n == NULL)
        || (r->e == NULL && e == NULL))
        return 0;

    if (n != NULL) {
        BN_free(r->n);
        r->n = n;
    }
    if (e != NULL) {
        BN_free(r->e);
        r->e = e;
    }
    if (d != NULL) {
        BN_clear_free(r->d);
        r->d = d;
    }

    return 1;
}

int RSA_set0_factors(RSA *r, BIGNUM *p, BIGNUM *q)
{
    /* If the fields p and q in r are NULL, the corresponding input
     * parameters MUST be non-NULL.
     */
    if ((r->p == NULL && p == NULL)
        || (r->q == NULL && q == NULL))
        return 0;

    if (p != NULL) {
        BN_clear_free(r->p);
        r->p = p;
    }
    if (q != NULL) {
        BN_clear_free(r->q);
        r->q = q;
    }

    return 1;
}

int RSA_set0_crt_params(RSA *r, BIGNUM *dmp1, BIGNUM *dmq1, BIGNUM *iqmp)
{
    /* If the fields dmp1, dmq1 and iqmp in r are NULL, the corresponding input
     * parameters MUST be non-NULL.
     */
    if ((r->dmp1 == NULL && dmp1 == NULL)
        || (r->dmq1 == NULL && dmq1 == NULL)
        || (r->iqmp == NULL && iqmp == NULL))
        return 0;

    if (dmp1 != NULL) {
        BN_clear_free(r->dmp1);
        r->dmp1 = dmp1;
    }
    if (dmq1 != NULL) {
        BN_clear_free(r->dmq1);
        r->dmq1 = dmq1;
    }
    if (iqmp != NULL) {
        BN_clear_free(r->iqmp);
        r->iqmp = iqmp;
    }

    return 1;
}

void RSA_get0_key(const RSA *r,
                  const BIGNUM **n, const BIGNUM **e, const BIGNUM **d)
{
    if (n != NULL)
        *n = r->n;
    if (e != NULL)
        *e = r->e;
    if (d != NULL)
        *d = r->d;
}

void RSA_get0_factors(const RSA *r, const BIGNUM **p, const BIGNUM **q)
{
    if (p != NULL)
        *p = r->p;
    if (q != NULL)
        *q = r->q;
}

void RSA_get0_crt_params(const RSA *r,
                         const BIGNUM **dmp1, const BIGNUM **dmq1,
                         const BIGNUM **iqmp)
{
    if (dmp1 != NULL)
        *dmp1 = r->dmp1;
    if (dmq1 != NULL)
        *dmq1 = r->dmq1;
    if (iqmp != NULL)
        *iqmp = r->iqmp;
}

void RSA_clear_flags(RSA *r, int flags)
{
    r->flags &= ~flags;
}

int RSA_test_flags(const RSA *r, int flags)
{
    return r->flags & flags;
}

void RSA_set_flags(RSA *r, int flags)
{
    r->flags |= flags;
}

ENGINE *RSA_get0_engine(const RSA *r)
{
    return r->engine;
}

#endif

#if OPENSSL_VERSION_NUMBER < 0x10101000L

const BIGNUM *RSA_get0_n(const RSA *r)
{
    const BIGNUM *n;
    RSA_get0_key(r, &n, NULL, NULL);
    return n;
}

const BIGNUM *RSA_get0_e(const RSA *r)
{
    const BIGNUM *e;
    RSA_get0_key(r, NULL, &e, NULL);
    return e;
}

const BIGNUM *RSA_get0_d(const RSA *r)
{
    const BIGNUM *d;
    RSA_get0_key(r, NULL, NULL, &d);
    return d;
}

const BIGNUM *RSA_get0_p(const RSA *r)
{
    const BIGNUM *p;
    RSA_get0_factors(r, &p, NULL);
    return p;
}

const BIGNUM *RSA_get0_q(const RSA *r)
{
    const BIGNUM *q;
    RSA_get0_factors(r, NULL, &q);
    return q;
}

const BIGNUM *RSA_get0_dmp1(const RSA *r)
{
    const BIGNUM *dmp1;
    RSA_get0_crt_params(r, &dmp1, NULL, NULL);
    return dmp1;
}

const BIGNUM *RSA_get0_dmq1(const RSA *r)
{
    const BIGNUM *dmq1;
    RSA_get0_crt_params(r, NULL, &dmq1, NULL);
    return dmq1;
}

const BIGNUM *RSA_get0_iqmp(const RSA *r)
{
    const BIGNUM *iqmp;
    RSA_get0_crt_params(r, NULL, NULL, &iqmp);
    return iqmp;
}

#endif
