#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "glibc.h"
#include "features.h"

namespace {
    void ReadAuxVector(Elf64_auxv_t** begin, Elf64_auxv_t** end) noexcept {
        int fd = open("/proc/self/auxv", O_RDONLY | O_CLOEXEC);
        if (fd == -1) {
            return;
        }

        constexpr size_t item_size = sizeof(Elf64_auxv_t);
        constexpr size_t block_size = item_size * 32;

        size_t bytes_read = 0;
        size_t size = 0;

        struct TBuffer {
            ~TBuffer() {
                free(Pointer);
            }
            char* Pointer = nullptr;
        } buffer;

        while (true) {
            size_t bytes_left = size - bytes_read;

            if (!bytes_left) {
                size += block_size;
                char* new_buffer = (char*)realloc(buffer.Pointer, size);
                if (!new_buffer) {
                    return;
                }
                buffer.Pointer = new_buffer;
                continue;
            }

            ssize_t r = read(fd, buffer.Pointer + bytes_read, bytes_left);
            if (!r) {
                break;
            } else if (r < 0) {
                if (errno == EINTR) {
                    continue;
                } else {
                    return;
                }
            }

            bytes_read += r;
        }

        size_t item_count = bytes_read / item_size;
        *begin = (Elf64_auxv_t*)buffer.Pointer;
        *end = (Elf64_auxv_t*)(buffer.Pointer + item_count * item_size);
        buffer.Pointer = nullptr;
    }
}

extern "C" {
    weak unsigned long __getauxval(unsigned long item);
}

namespace NUbuntuCompat {

    TGlibc::TGlibc() noexcept
        : AuxVectorBegin(nullptr)
        , AuxVectorEnd(nullptr)
    {
        if (!__getauxval) {
            ReadAuxVector((Elf64_auxv_t**)&AuxVectorBegin, (Elf64_auxv_t**)&AuxVectorEnd);
        }

        Secure = (bool)GetAuxVal(AT_SECURE);
    }

    TGlibc::~TGlibc() noexcept {
        free(AuxVectorBegin);
    }

    unsigned long TGlibc::GetAuxVal(unsigned long item) noexcept {
        if (__getauxval) {
            return __getauxval(item);
        }

        for (Elf64_auxv_t* p = (Elf64_auxv_t*)AuxVectorBegin; p < (Elf64_auxv_t*)AuxVectorEnd; ++p) {
            if (p->a_type == item) {
                return p->a_un.a_val;
            }
        }

        errno = ENOENT;
        return 0;
    }

    bool TGlibc::IsSecure() noexcept {
        return Secure;
    }

    static TGlibc __attribute__((__init_priority__(101))) GlibcInstance;

    TGlibc& GetGlibc() noexcept {
        return GlibcInstance;
    }
}
