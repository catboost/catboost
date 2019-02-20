#include "x509.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

const ASN1_TIME *X509_get0_notBefore(const X509 *x)
{
    return x->cert_info->validity->notBefore;
}

const ASN1_TIME *X509_get0_notAfter(const X509 *x)
{
    return x->cert_info->validity->notAfter;
}

ASN1_TIME *X509_getm_notBefore(const X509 *x)
{
    return x->cert_info->validity->notBefore;
}

ASN1_TIME *X509_getm_notAfter(const X509 *x)
{
    return x->cert_info->validity->notAfter;
}

#endif
