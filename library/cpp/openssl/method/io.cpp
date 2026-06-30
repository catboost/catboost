#include "io.h"

#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

namespace {
    using NOpenSSL::TAbstractIO;

    TAbstractIO* IO(BIO* bio) noexcept {
        void* ptr = BIO_get_data(bio);
        Y_ABORT_UNLESS(ptr);
        return static_cast<TAbstractIO*>(ptr);
    }

    template<class T, class Callable, class... Args>
    T ExceptionBoundary(BIO* bio, Callable&& f, T err, Args&&... args) noexcept {
        try {
            return (IO(bio)->*f)(args...);
        } catch (...) {
            return err;
        }
    }

    int Write(BIO* bio, const char* data, int dlen) noexcept {
        return ExceptionBoundary(bio, &TAbstractIO::WriteOld, -1, data, dlen);
    }

    int Read(BIO* bio, char* data, int dlen) noexcept {
        return ExceptionBoundary(bio, &TAbstractIO::ReadOld, -1, data, dlen);
    }

    int Puts(BIO* bio, const char* buf) noexcept {
        return ExceptionBoundary(bio, &TAbstractIO::Puts, -1, buf);
    }

    int Gets(BIO* bio, char* buf, int size) noexcept {
        return ExceptionBoundary(bio, &TAbstractIO::Gets, -1, buf, size);
    }

    long Ctrl(BIO* bio, int cmd, long larg, void* parg) noexcept {
        return ExceptionBoundary(bio, &TAbstractIO::Ctrl, -1, cmd, larg, parg);
    }

    int Create(BIO* bio) noexcept {
        BIO_set_data(bio, nullptr);
        BIO_set_init(bio, 1);
        return 1;
    }

    int Destroy(BIO* bio) noexcept {
        BIO_set_data(bio, nullptr);
        BIO_set_init(bio, 0);
        return 1;
    }

    NOpenSSL::TBioMethod* Method() {
        return SingletonWithPriority<NOpenSSL::TBioMethod, 32768>(
            BIO_get_new_index() | BIO_TYPE_SOURCE_SINK,
            "AbstractIO",
            Write,
            Read,
            Puts,
            Gets,
            Ctrl,
            Create,
            Destroy,
            nullptr
        );
    }
}

namespace NOpenSSL {

    TAbstractIO::TAbstractIO()
        : Bio(BIO_new(*Method())) {
        if (Y_UNLIKELY(!Bio)) {
            throw std::bad_alloc();
        }
        BIO_set_data(Bio, this);
    }

    TAbstractIO::~TAbstractIO() {
        BIO_free(Bio);
    }

    int TAbstractIO::WriteOld(const char* data, int dlen) {
        size_t written = 0;

        int ret = Write(data, dlen, &written);
        if (ret <= 0) {
            return ret;
        }

        return written;
    }

    int TAbstractIO::ReadOld(char* data, int dlen) {
        size_t readbytes = 0;

        int ret = Read(data, dlen, &readbytes);
        if (ret <= 0) {
            return ret;
        }

        return readbytes;
    }

    long TAbstractIO::Ctrl(int cmd, long larg, void* parg) {
        Y_UNUSED(larg);
        Y_UNUSED(parg);

        if (cmd == BIO_CTRL_FLUSH) {
            Flush();
            return 1;
        }

        return 0;
    }

} // namespace NOpenSSL
