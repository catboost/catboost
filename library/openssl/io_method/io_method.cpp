#include "io_method.h"

#include <util/generic/utility.h>
#include <util/system/yassert.h>

namespace {

    BIO_METHOD* BIO_meth_new(int type, const char* name) noexcept {
        BIO_METHOD* biom = static_cast<BIO_METHOD*>(OPENSSL_malloc(sizeof(BIO_METHOD)));
        if (biom == nullptr) {
            return nullptr;
        }

        Zero(*biom);

        biom->type = type;
        biom->name = name;
        return biom;
    }

    void BIO_meth_free(BIO_METHOD* biom) noexcept {
        OPENSSL_free(biom);
    }

} // namespace

namespace NOpenSSL {

    TIOMethod::TIOMethod(
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
    ) noexcept {
        Method = BIO_meth_new(type, name);
        Y_VERIFY(Method, "Failed to allocate new BIO_METHOD");

        Method->bwrite = write;
        Method->bread = read;
        Method->bputs = puts;
        Method->bgets = gets;
        Method->ctrl = ctrl;
        Method->create = create;
        Method->destroy = destroy;
        Method->callback_ctrl = callbackCtrl;
    }

    TIOMethod::~TIOMethod() noexcept {
        BIO_meth_free(Method);
    }

} // namespace NOpenSSL
