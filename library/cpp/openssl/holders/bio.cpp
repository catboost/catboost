#include "bio.h"

namespace NOpenSSL {

    TBioMethod::TBioMethod(
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
    )
        : THolder(type, name)
    {
        BIO_meth_set_write(*this, write);
        BIO_meth_set_read(*this, read);
        BIO_meth_set_puts(*this, puts);
        BIO_meth_set_gets(*this, gets);
        BIO_meth_set_ctrl(*this, ctrl);
        BIO_meth_set_create(*this, create);
        BIO_meth_set_destroy(*this, destroy);
        BIO_meth_set_callback_ctrl(*this, callbackCtrl);
    }

} // namespace NOpenSSL
