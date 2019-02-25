#include "bio.h"

#include "crypto.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

int BIO_get_new_index()
{
    static int bio_count = BIO_TYPE_START;

    return CRYPTO_add(&bio_count, 1, CRYPTO_LOCK_BIO);
}

void BIO_set_data(BIO *a, void *ptr)
{
    a->ptr = ptr;
}

void *BIO_get_data(BIO *a)
{
    return a->ptr;
}

void BIO_set_init(BIO *a, int init)
{
    a->init = init;
}

int BIO_get_init(BIO *a)
{
    return a->init;
}

void BIO_set_shutdown(BIO *a, int shut)
{
    a->shutdown = shut;
}

int BIO_get_shutdown(BIO *a)
{
    return a->shutdown;
}

int BIO_up_ref(BIO *a)
{
    CRYPTO_add(&a->references, 1, CRYPTO_LOCK_BIO);
    return 1;
}

BIO_METHOD *BIO_meth_new(int type, const char *name)
{
    BIO_METHOD *biom = OPENSSL_zalloc(sizeof(BIO_METHOD));

    if (biom == NULL) {
        return NULL;
    }

    biom->type = type;
    biom->name = name;
    return biom;
}

void BIO_meth_free(BIO_METHOD *biom)
{
    OPENSSL_free(biom);
}

int (*BIO_meth_get_write(const BIO_METHOD *biom)) (BIO *, const char *, int)
{
    return biom->bwrite;
}

int BIO_meth_set_write(BIO_METHOD *biom,
                       int (*bwrite) (BIO *, const char *, int))
{
    biom->bwrite = bwrite;
    return 1;
}

int (*BIO_meth_get_read(const BIO_METHOD *biom)) (BIO *, char *, int)
{
    return biom->bread;
}

int BIO_meth_set_read(BIO_METHOD *biom,
                      int (*bread) (BIO *, char *, int))
{
    biom->bread = bread;
    return 1;
}

int (*BIO_meth_get_puts(const BIO_METHOD *biom)) (BIO *, const char *)
{
    return biom->bputs;
}

int BIO_meth_set_puts(BIO_METHOD *biom,
                      int (*bputs) (BIO *, const char *))
{
    biom->bputs = bputs;
    return 1;
}

int (*BIO_meth_get_gets(const BIO_METHOD *biom)) (BIO *, char *, int)
{
    return biom->bgets;
}

int BIO_meth_set_gets(BIO_METHOD *biom,
                      int (*bgets) (BIO *, char *, int))
{
    biom->bgets = bgets;
    return 1;
}

long (*BIO_meth_get_ctrl(const BIO_METHOD *biom)) (BIO *, int, long, void *)
{
    return biom->ctrl;
}

int BIO_meth_set_ctrl(BIO_METHOD *biom,
                      long (*ctrl) (BIO *, int, long, void *))
{
    biom->ctrl = ctrl;
    return 1;
}

int (*BIO_meth_get_create(const BIO_METHOD *biom)) (BIO *)
{
    return biom->create;
}

int BIO_meth_set_create(BIO_METHOD *biom, int (*create) (BIO *))
{
    biom->create = create;
    return 1;
}

int (*BIO_meth_get_destroy(const BIO_METHOD *biom)) (BIO *)
{
    return biom->destroy;
}

int BIO_meth_set_destroy(BIO_METHOD *biom, int (*destroy) (BIO *))
{
    biom->destroy = destroy;
    return 1;
}

long (*BIO_meth_get_callback_ctrl(const BIO_METHOD *biom)) (BIO *, int, BIO_info_cb *)
{
    return biom->callback_ctrl;
}

int BIO_meth_set_callback_ctrl(BIO_METHOD *biom,
                               long (*callback_ctrl) (BIO *, int,
                                                      BIO_info_cb *))
{
    biom->callback_ctrl = callback_ctrl;
    return 1;
}

#endif
