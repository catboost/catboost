#pragma once

#include <contrib/libs/openssl/include/openssl/bn.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

int BN_security_bits(int L, int N);

#endif

#ifdef  __cplusplus
}
#endif
