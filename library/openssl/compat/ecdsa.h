#pragma once

#include <contrib/libs/openssl/include/openssl/ecdsa.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

/** Accessor for r and s fields of ECDSA_SIG
 *  \param  sig  pointer to ECDSA_SIG pointer
 *  \param  pr   pointer to BIGNUM pointer for r (may be NULL)
 *  \param  ps   pointer to BIGNUM pointer for s (may be NULL)
 */
void ECDSA_SIG_get0(const ECDSA_SIG *sig, const BIGNUM **pr, const BIGNUM **ps);

/** Setter for r and s fields of ECDSA_SIG
 *  \param  sig  pointer to ECDSA_SIG pointer
 *  \param  r    pointer to BIGNUM for r (may be NULL)
 *  \param  s    pointer to BIGNUM for s (may be NULL)
 */
int ECDSA_SIG_set0(ECDSA_SIG *sig, BIGNUM *r, BIGNUM *s);

#endif

#if OPENSSL_VERSION_NUMBER < 0x10101000L

/** Accessor for r field of ECDSA_SIG
 *  \param  sig  pointer to ECDSA_SIG structure
 */
const BIGNUM *ECDSA_SIG_get0_r(const ECDSA_SIG *sig);

/** Accessor for s field of ECDSA_SIG
 *  \param  sig  pointer to ECDSA_SIG structure
 */
const BIGNUM *ECDSA_SIG_get0_s(const ECDSA_SIG *sig);

#endif

#ifdef  __cplusplus
}
#endif
