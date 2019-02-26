#include "dsa.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

void DSA_get0_pqg(const DSA *d,
                  const BIGNUM **p, const BIGNUM **q, const BIGNUM **g)
{
    if (p != NULL)
        *p = d->p;
    if (q != NULL)
        *q = d->q;
    if (g != NULL)
        *g = d->g;
}

int DSA_set0_pqg(DSA *d, BIGNUM *p, BIGNUM *q, BIGNUM *g)
{
    /* If the fields p, q and g in d are NULL, the corresponding input
     * parameters MUST be non-NULL.
     */
    if ((d->p == NULL && p == NULL)
        || (d->q == NULL && q == NULL)
        || (d->g == NULL && g == NULL))
        return 0;

    if (p != NULL) {
        BN_free(d->p);
        d->p = p;
    }
    if (q != NULL) {
        BN_free(d->q);
        d->q = q;
    }
    if (g != NULL) {
        BN_free(d->g);
        d->g = g;
    }

    return 1;
}

void DSA_get0_key(const DSA *d,
                  const BIGNUM **pub_key, const BIGNUM **priv_key)
{
    if (pub_key != NULL)
        *pub_key = d->pub_key;
    if (priv_key != NULL)
        *priv_key = d->priv_key;
}

int DSA_set0_key(DSA *d, BIGNUM *pub_key, BIGNUM *priv_key)
{
    /* If the field pub_key in d is NULL, the corresponding input
     * parameters MUST be non-NULL.  The priv_key field may
     * be left NULL.
     */
    if (d->pub_key == NULL && pub_key == NULL)
        return 0;

    if (pub_key != NULL) {
        BN_free(d->pub_key);
        d->pub_key = pub_key;
    }
    if (priv_key != NULL) {
        BN_free(d->priv_key);
        d->priv_key = priv_key;
    }

    return 1;
}

void DSA_clear_flags(DSA *d, int flags)
{
    d->flags &= ~flags;
}

int DSA_test_flags(const DSA *d, int flags)
{
    return d->flags & flags;
}

void DSA_set_flags(DSA *d, int flags)
{
    d->flags |= flags;
}

ENGINE *DSA_get0_engine(DSA *d)
{
    return d->engine;
}

#endif

#if OPENSSL_VERSION_NUMBER < 0x10101000L

const BIGNUM *DSA_get0_p(const DSA *d)
{
    const BIGNUM *p;
    DSA_get0_pqg(d, &p, NULL, NULL);
    return p;
}

const BIGNUM *DSA_get0_q(const DSA *d)
{
    const BIGNUM *q;
    DSA_get0_pqg(d, NULL, &q, NULL);
    return q;
}

const BIGNUM *DSA_get0_g(const DSA *d)
{
    const BIGNUM *g;
    DSA_get0_pqg(d, NULL, NULL, &g);
    return g;
}

const BIGNUM *DSA_get0_pub_key(const DSA *d)
{
    const BIGNUM *pub_key;
    DSA_get0_key(d, &pub_key, NULL);
    return pub_key;
}

const BIGNUM *DSA_get0_priv_key(const DSA *d)
{
    const BIGNUM *priv_key;
    DSA_get0_key(d, NULL, &priv_key);
    return priv_key;
}

#endif
