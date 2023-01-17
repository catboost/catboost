#pragma once

#include <openssl/x509_vfy.h>

#include <library/cpp/openssl/holders/holder.h>

namespace NOpenSSL {

class TX509LookupMethod : public THolder<X509_LOOKUP_METHOD, X509_LOOKUP_meth_new, X509_LOOKUP_meth_free, const char*> {
public:
    TX509LookupMethod(
        const char* name,
        int (*newItem) (X509_LOOKUP *ctx),
        void (*free) (X509_LOOKUP *ctx),
        int (*init) (X509_LOOKUP *ctx),
        int (*shutdown) (X509_LOOKUP *ctx),
        X509_LOOKUP_ctrl_fn ctrl,
        X509_LOOKUP_get_by_subject_fn getBySubject,
        X509_LOOKUP_get_by_issuer_serial_fn getByIssuerSerial,
        X509_LOOKUP_get_by_fingerprint_fn getByFingerprint,
        X509_LOOKUP_get_by_alias_fn getByAlias
    );
};

} // namespace NOpenSSL
