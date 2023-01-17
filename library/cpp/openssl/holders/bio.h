#pragma once

#include <openssl/bio.h>

#include <library/cpp/openssl/holders/holder.h>

namespace NOpenSSL {

class TBioMethod : public THolder<BIO_METHOD, BIO_meth_new, BIO_meth_free, int, const char*> {
public:
    TBioMethod(
        int type,
        const char* name,
        int (*write)(BIO*, const char*, int),
        int (*read)(BIO*, char*, int),
        int (*puts)(BIO*, const char*),
        int (*gets)(BIO*, char*, int),
        long (*ctrl)(BIO*, int, long, void*),
        int (*create)(BIO*),
        int (*destroy)(BIO*),
        long (*callbackCtrl)(BIO*, int, bio_info_cb*)
    );
};

} // namespace NOpenSSL
