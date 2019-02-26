#include "x509.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

const ASN1_INTEGER *X509_get0_serialNumber(const X509 *a)
{
    return a->cert_info->serialNumber;
}

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

int X509_up_ref(X509 *x)
{
    CRYPTO_add(&x->references, 1, CRYPTO_LOCK_X509);
    return 1;
}

int X509_CRL_up_ref(X509_CRL *crl)
{
    CRYPTO_add(&crl->references, 1, CRYPTO_LOCK_X509_CRL);
    return 1;
}

#endif
