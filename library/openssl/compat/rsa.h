#pragma once

#include <contrib/libs/openssl/include/openssl/rsa.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

int RSA_bits(const RSA *r);
int RSA_security_bits(const RSA *rsa);

int RSA_set0_key(RSA *r, BIGNUM *n, BIGNUM *e, BIGNUM *d);
int RSA_set0_factors(RSA *r, BIGNUM *p, BIGNUM *q);
int RSA_set0_crt_params(RSA *r, BIGNUM *dmp1, BIGNUM *dmq1, BIGNUM *iqmp);
void RSA_get0_key(const RSA *r,
                  const BIGNUM **n, const BIGNUM **e, const BIGNUM **d);
void RSA_get0_factors(const RSA *r, const BIGNUM **p, const BIGNUM **q);
void RSA_get0_crt_params(const RSA *r,
                         const BIGNUM **dmp1, const BIGNUM **dmq1,
                         const BIGNUM **iqmp);
void RSA_clear_flags(RSA *r, int flags);
int RSA_test_flags(const RSA *r, int flags);
void RSA_set_flags(RSA *r, int flags);
ENGINE *RSA_get0_engine(const RSA *r);

#endif

#if OPENSSL_VERSION_NUMBER < 0x10101000L

const BIGNUM *RSA_get0_n(const RSA *r);
const BIGNUM *RSA_get0_e(const RSA *r);
const BIGNUM *RSA_get0_d(const RSA *r);
const BIGNUM *RSA_get0_p(const RSA *r);
const BIGNUM *RSA_get0_q(const RSA *r);
const BIGNUM *RSA_get0_dmp1(const RSA *r);
const BIGNUM *RSA_get0_dmq1(const RSA *r);
const BIGNUM *RSA_get0_iqmp(const RSA *r);

#endif

#ifdef  __cplusplus
}
#endif
