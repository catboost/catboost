#pragma once

#include <contrib/libs/openssl/include/openssl/bio.h>

namespace NOpenSSL {

class TIOMethod {
public:
    TIOMethod(
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
    ) noexcept;

    TIOMethod(const TIOMethod&) = delete;
    TIOMethod& operator=(const TIOMethod&) = delete;

    ~TIOMethod() noexcept;

    inline operator BIO_METHOD* () noexcept {
        return Method;
    }

private:
    BIO_METHOD* Method;
};

} // namespace NOpenSSL
