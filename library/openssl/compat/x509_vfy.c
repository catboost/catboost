#include "x509_vfy.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L

#include "crypto.h"
#include "x509.h"

X509 *X509_OBJECT_get0_X509(const X509_OBJECT *a)
{
    if (a == NULL || a->type != X509_LU_X509)
        return NULL;
    return a->data.x509;
}

X509_CRL *X509_OBJECT_get0_X509_CRL(X509_OBJECT *a)
{
    if (a == NULL || a->type != X509_LU_CRL)
        return NULL;
    return a->data.crl;
}

X509_LOOKUP_TYPE X509_OBJECT_get_type(const X509_OBJECT *a)
{
    return a->type;
}

X509_OBJECT *X509_OBJECT_new()
{
    X509_OBJECT *ret = OPENSSL_zalloc(sizeof(*ret));

    if (ret == NULL) {
        return NULL;
    }
    ret->type = X509_LU_NONE;
    return ret;
}

static void x509_object_free_internal(X509_OBJECT *a)
{
    if (a == NULL)
        return;
    switch (a->type) {
    default:
        break;
    case X509_LU_X509:
        X509_free(a->data.x509);
        break;
    case X509_LU_CRL:
        X509_CRL_free(a->data.crl);
        break;
    }
}
int X509_OBJECT_set1_X509(X509_OBJECT *a, X509 *obj)
{
    if (a == NULL || !X509_up_ref(obj))
        return 0;

    x509_object_free_internal(a);
    a->type = X509_LU_X509;
    a->data.x509 = obj;
    return 1;
}

int X509_OBJECT_set1_X509_CRL(X509_OBJECT *a, X509_CRL *obj)
{
    if (a == NULL || !X509_CRL_up_ref(obj))
        return 0;

    x509_object_free_internal(a);
    a->type = X509_LU_CRL;
    a->data.crl = obj;
    return 1;
}

void X509_OBJECT_free(X509_OBJECT *a)
{
    x509_object_free_internal(a);
    OPENSSL_free(a);
}

X509_LOOKUP_METHOD *X509_LOOKUP_meth_new(const char *name)
{
    X509_LOOKUP_METHOD *method = OPENSSL_zalloc(sizeof(X509_LOOKUP_METHOD));

    if (method == NULL) {
        return NULL;
    }

    method->name = name;
    return method;
}

void X509_LOOKUP_meth_free(X509_LOOKUP_METHOD *method)
{
    OPENSSL_free(method);
}

int X509_LOOKUP_meth_set_new_item(X509_LOOKUP_METHOD *method,
                                  int (*new_item) (X509_LOOKUP *ctx))
{
    method->new_item = new_item;
    return 1;
}

int (*X509_LOOKUP_meth_get_new_item(const X509_LOOKUP_METHOD* method))
    (X509_LOOKUP *ctx)
{
    return method->new_item;
}

int X509_LOOKUP_meth_set_free(
    X509_LOOKUP_METHOD *method,
    void (*free) (X509_LOOKUP *ctx))
{
    method->free = free;
    return 1;
}

void (*X509_LOOKUP_meth_get_free(const X509_LOOKUP_METHOD* method))
    (X509_LOOKUP *ctx)
{
    return method->free;
}

int X509_LOOKUP_meth_set_init(X509_LOOKUP_METHOD *method,
                              int (*init) (X509_LOOKUP *ctx))
{
    method->init = init;
    return 1;
}

int (*X509_LOOKUP_meth_get_init(const X509_LOOKUP_METHOD* method))
    (X509_LOOKUP *ctx)
{
    return method->init;
}

int X509_LOOKUP_meth_set_shutdown(
    X509_LOOKUP_METHOD *method,
    int (*shutdown) (X509_LOOKUP *ctx))
{
    method->shutdown = shutdown;
    return 1;
}

int (*X509_LOOKUP_meth_get_shutdown(const X509_LOOKUP_METHOD* method))
    (X509_LOOKUP *ctx)
{
    return method->shutdown;
}

int X509_LOOKUP_meth_set_ctrl(
    X509_LOOKUP_METHOD *method,
    X509_LOOKUP_ctrl_fn ctrl)
{
    method->ctrl = ctrl;
    return 1;
}

X509_LOOKUP_ctrl_fn X509_LOOKUP_meth_get_ctrl(const X509_LOOKUP_METHOD *method)
{
    return method->ctrl;
}

int X509_LOOKUP_meth_set_get_by_subject(X509_LOOKUP_METHOD *method,
    X509_LOOKUP_get_by_subject_fn get_by_subject)
{
    method->get_by_subject = get_by_subject;
    return 1;
}

X509_LOOKUP_get_by_subject_fn X509_LOOKUP_meth_get_get_by_subject(
    const X509_LOOKUP_METHOD *method)
{
    return method->get_by_subject;
}


int X509_LOOKUP_meth_set_get_by_issuer_serial(X509_LOOKUP_METHOD *method,
    X509_LOOKUP_get_by_issuer_serial_fn get_by_issuer_serial)
{
    method->get_by_issuer_serial = get_by_issuer_serial;
    return 1;
}

X509_LOOKUP_get_by_issuer_serial_fn
    X509_LOOKUP_meth_get_get_by_issuer_serial(const X509_LOOKUP_METHOD *method)
{
    return method->get_by_issuer_serial;
}


int X509_LOOKUP_meth_set_get_by_fingerprint(X509_LOOKUP_METHOD *method,
    X509_LOOKUP_get_by_fingerprint_fn get_by_fingerprint)
{
    method->get_by_fingerprint = get_by_fingerprint;
    return 1;
}

X509_LOOKUP_get_by_fingerprint_fn X509_LOOKUP_meth_get_get_by_fingerprint(
    const X509_LOOKUP_METHOD *method)
{
    return method->get_by_fingerprint;
}

int X509_LOOKUP_meth_set_get_by_alias(X509_LOOKUP_METHOD *method,
                                      X509_LOOKUP_get_by_alias_fn get_by_alias)
{
    method->get_by_alias = get_by_alias;
    return 1;
}

X509_LOOKUP_get_by_alias_fn X509_LOOKUP_meth_get_get_by_alias(
    const X509_LOOKUP_METHOD *method)
{
    return method->get_by_alias;
}

int X509_LOOKUP_set_method_data(X509_LOOKUP *ctx, void *data)
{
    ctx->method_data = data;
    return 1;
}

void *X509_LOOKUP_get_method_data(const X509_LOOKUP *ctx)
{
    return ctx->method_data;
}

#endif
