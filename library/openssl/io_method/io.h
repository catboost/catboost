#pragma once

#include "io_method.h"

#include <util/generic/yexception.h>
#include <util/system/compiler.h>

namespace NOpenSSL {

class TAbstractIO {
public:
    TAbstractIO();
    virtual ~TAbstractIO();

    virtual int Write(const char* data, size_t dlen, size_t* written) = 0;
    virtual int Read(char* data, size_t dlen, size_t* readbytes) = 0;
    virtual int Puts(const char* buf) = 0;
    virtual int Gets(char* buf, int size) = 0;

    virtual long Ctrl(int cmd, long larg, void* parg);
    virtual void Flush() = 0;

    int WriteOld(const char* data, int dlen);
    int ReadOld(char* data, int dlen);

    inline operator BIO* () noexcept {
        return Bio;
    }

private:
    static TIOMethod Method;

    BIO* Bio;
};

} // namespace NOpenSSL
