#pragma once

#include <contrib/libs/openssl/include/openssl/x509_vfy.h>

#ifdef  __cplusplus
extern "C" {
#endif

#if OPENSSL_VERSION_NUMBER < 0x10100000L

#define X509_LU_NONE 0

typedef int X509_LOOKUP_TYPE;

X509_OBJECT *X509_OBJECT_new(void);
void X509_OBJECT_free(X509_OBJECT *a);
X509_LOOKUP_TYPE X509_OBJECT_get_type(const X509_OBJECT *a);
X509 *X509_OBJECT_get0_X509(const X509_OBJECT *a);
int X509_OBJECT_set1_X509(X509_OBJECT *a, X509 *obj);
X509_CRL *X509_OBJECT_get0_X509_CRL(X509_OBJECT *a);
int X509_OBJECT_set1_X509_CRL(X509_OBJECT *a, X509_CRL *obj);

typedef int (*X509_LOOKUP_ctrl_fn)(X509_LOOKUP *ctx, int cmd, const char *argc,
                                   long argl, char **ret);
typedef int (*X509_LOOKUP_get_by_subject_fn)(X509_LOOKUP *ctx,
                                             X509_LOOKUP_TYPE type,
                                             X509_NAME *name,
                                             X509_OBJECT *ret);
typedef int (*X509_LOOKUP_get_by_issuer_serial_fn)(X509_LOOKUP *ctx,
                                                   X509_LOOKUP_TYPE type,
                                                   X509_NAME *name,
                                                   ASN1_INTEGER *serial,
                                                   X509_OBJECT *ret);
typedef int (*X509_LOOKUP_get_by_fingerprint_fn)(X509_LOOKUP *ctx,
                                                 X509_LOOKUP_TYPE type,
                                                 unsigned char* bytes,
                                                 int len,
                                                 X509_OBJECT *ret);
typedef int (*X509_LOOKUP_get_by_alias_fn)(X509_LOOKUP *ctx,
                                           X509_LOOKUP_TYPE type,
                                           char *str,
                                           int len,
                                           X509_OBJECT *ret);

X509_LOOKUP_METHOD *X509_LOOKUP_meth_new(const char *name);
void X509_LOOKUP_meth_free(X509_LOOKUP_METHOD *method);

int X509_LOOKUP_meth_set_new_item(X509_LOOKUP_METHOD *method,
                                  int (*new_item) (X509_LOOKUP *ctx));
int (*X509_LOOKUP_meth_get_new_item(const X509_LOOKUP_METHOD* method))
    (X509_LOOKUP *ctx);

int X509_LOOKUP_meth_set_free(X509_LOOKUP_METHOD *method,
                              void (*free) (X509_LOOKUP *ctx));
void (*X509_LOOKUP_meth_get_free(const X509_LOOKUP_METHOD* method))
    (X509_LOOKUP *ctx);

int X509_LOOKUP_meth_set_init(X509_LOOKUP_METHOD *method,
                              int (*init) (X509_LOOKUP *ctx));
int (*X509_LOOKUP_meth_get_init(const X509_LOOKUP_METHOD* method))
    (X509_LOOKUP *ctx);

int X509_LOOKUP_meth_set_shutdown(X509_LOOKUP_METHOD *method,
                                  int (*shutdown) (X509_LOOKUP *ctx));
int (*X509_LOOKUP_meth_get_shutdown(const X509_LOOKUP_METHOD* method))
    (X509_LOOKUP *ctx);

int X509_LOOKUP_meth_set_ctrl(X509_LOOKUP_METHOD *method,
                              X509_LOOKUP_ctrl_fn ctrl_fn);
X509_LOOKUP_ctrl_fn X509_LOOKUP_meth_get_ctrl(const X509_LOOKUP_METHOD *method);

int X509_LOOKUP_meth_set_get_by_subject(X509_LOOKUP_METHOD *method,
                                        X509_LOOKUP_get_by_subject_fn fn);
X509_LOOKUP_get_by_subject_fn X509_LOOKUP_meth_get_get_by_subject(
    const X509_LOOKUP_METHOD *method);

int X509_LOOKUP_meth_set_get_by_issuer_serial(X509_LOOKUP_METHOD *method,
    X509_LOOKUP_get_by_issuer_serial_fn fn);
X509_LOOKUP_get_by_issuer_serial_fn X509_LOOKUP_meth_get_get_by_issuer_serial(
    const X509_LOOKUP_METHOD *method);

int X509_LOOKUP_meth_set_get_by_fingerprint(X509_LOOKUP_METHOD *method,
    X509_LOOKUP_get_by_fingerprint_fn fn);
X509_LOOKUP_get_by_fingerprint_fn X509_LOOKUP_meth_get_get_by_fingerprint(
    const X509_LOOKUP_METHOD *method);

int X509_LOOKUP_meth_set_get_by_alias(X509_LOOKUP_METHOD *method,
                                      X509_LOOKUP_get_by_alias_fn fn);
X509_LOOKUP_get_by_alias_fn X509_LOOKUP_meth_get_get_by_alias(
    const X509_LOOKUP_METHOD *method);

int X509_LOOKUP_set_method_data(X509_LOOKUP *ctx, void *data);
void *X509_LOOKUP_get_method_data(const X509_LOOKUP *ctx);

#endif

#ifdef  __cplusplus
}
#endif
