#pragma once

#include <contrib/libs/openssl/include/openssl/ssl.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

void SSL_set0_rbio(SSL *s, BIO *rbio);
void SSL_set0_wbio(SSL *s, BIO *wbio);

pem_password_cb *SSL_CTX_get_default_passwd_cb(SSL_CTX *ctx);
void *SSL_CTX_get_default_passwd_cb_userdata(SSL_CTX *ctx);

#endif

#ifdef  __cplusplus
}
#endif
