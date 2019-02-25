#pragma once

#include <contrib/libs/openssl/include/openssl/asn1.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

const unsigned char *ASN1_STRING_get0_data(const ASN1_STRING *x);

#endif

#ifdef  __cplusplus
}
#endif
