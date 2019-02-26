#include "ssl.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

void SSL_set0_rbio(SSL *s, BIO *rbio)
{
    if ((s->rbio != NULL) && (s->rbio != rbio))
        BIO_free_all(s->rbio);
    s->rbio = rbio;
}

void SSL_set0_wbio(SSL *s, BIO *wbio)
{
    /*
     * If the output buffering BIO is still in place, remove it
     */
    if (s->bbio != NULL) {
        if (s->wbio == s->bbio) {
            s->wbio = s->wbio->next_bio;
            s->bbio->next_bio = NULL;
        }
    }
    if ((s->wbio != NULL) && (s->wbio != wbio))
        BIO_free_all(s->wbio);
    s->wbio = wbio;
}

pem_password_cb *SSL_CTX_get_default_passwd_cb(SSL_CTX *ctx)
{
    return ctx->default_passwd_callback;
}

void *SSL_CTX_get_default_passwd_cb_userdata(SSL_CTX *ctx)
{
    return ctx->default_passwd_callback_userdata;
}

#endif
