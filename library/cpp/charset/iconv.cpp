#include "iconv.h"

#include <iconv.h>

using namespace NICONVPrivate;

TDescriptor::TDescriptor(const char* from, const char* to)
    : Descriptor_(iconv_open(to, from))
    , From_(from)
    , To_(to)
{
#if defined(USE_ICONV_EXTENSIONS)
    if (!Invalid()) {
        int temp = 1;

        libiconvctl(Descriptor_, ICONV_SET_DISCARD_ILSEQ, &temp);
    }
#endif
}

TDescriptor::~TDescriptor() {
    if (!Invalid()) {
        iconv_close(static_cast<iconv_t>(Descriptor_));
    }
}

size_t NICONVPrivate::RecodeImpl(const TDescriptor& descriptor, const char* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written) {
    Y_ASSERT(!descriptor.Invalid());
    Y_ASSERT(in);
    Y_ASSERT(out);

    char* inPtr = const_cast<char*>(in);
    char* outPtr = out;
    size_t inSizeMod = inSize;
    size_t outSizeMod = outSize;
    size_t res = iconv(static_cast<iconv_t>(descriptor.Get()), &inPtr, &inSizeMod, &outPtr, &outSizeMod);

    read = inSize - inSizeMod;
    written = outSize - outSizeMod;

    return res;
}

void NICONVPrivate::DoRecode(const TDescriptor& descriptor, const char* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written) {
    if (descriptor.Invalid()) {
        ythrow yexception() << "Can not convert from " << descriptor.From() << " to " << descriptor.To();
    }

    size_t res = RecodeImpl(descriptor, in, out, inSize, outSize, read, written);

    if (res == static_cast<size_t>(-1)) {
        switch (errno) {
            case EILSEQ:
                read = inSize;
                break;

            case EINVAL:
                read = inSize;
                break;

            case E2BIG:
                ythrow yexception() << "Iconv error: output buffer is too small";

            default:
                ythrow yexception() << "Unknown iconv error";
        }
    }
}

RECODE_RESULT NICONVPrivate::DoRecodeNoThrow(const TDescriptor& descriptor, const char* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written) {
    if (descriptor.Invalid()) {
        return RECODE_ERROR;
    }

    size_t res = RecodeImpl(descriptor, in, out, inSize, outSize, read, written);

    if (res == static_cast<size_t>(-1)) {
        switch (errno) {
            case EILSEQ:
                read = inSize;
                break;

            case EINVAL:
                read = inSize;
                break;

            case E2BIG:
                return RECODE_EOOUTPUT;

            default:
                return RECODE_ERROR;
        }
    }

    return RECODE_OK;
}
