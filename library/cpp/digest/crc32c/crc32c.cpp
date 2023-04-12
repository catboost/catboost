#include "crc32c.h"

#include <util/generic/singleton.h>

#include <contrib/libs/crcutil/interface.h>

namespace {
    typedef crcutil_interface::CRC TCrc;

    struct TCrcUtilSse4 {
        TCrc* const Pimpl;

        TCrcUtilSse4() noexcept
            : Pimpl(TCrc::Create(0x82f63b78, 0, 32, true, 0, 0, 0, TCrc::IsSSE42Available(), nullptr))
        {
        }

        ~TCrcUtilSse4() noexcept {
            Pimpl->Delete();
        }

        inline ui32 Extend(ui32 init, const void* data, size_t n) const noexcept {
            crcutil_interface::UINT64 sum = init;
            Pimpl->Compute(data, n, &sum);
            return (ui32)sum;
        }

        inline ui32 Combine(ui32 blockACrc, ui32 blockBCrc, size_t blockBSize) const noexcept {
            crcutil_interface::UINT64 sum = blockACrc;
            Pimpl->Concatenate(blockBCrc, 0, blockBSize, &sum);
            return (ui32)sum;
        }
    };
}

ui32 Crc32c(const void* p, size_t size) noexcept {
    return Singleton<TCrcUtilSse4>()->Extend(0, p, size);
}

ui32 Crc32cExtend(ui32 init, const void* data, size_t n) noexcept {
    return Singleton<TCrcUtilSse4>()->Extend(init, data, n);
}

ui32 Crc32cCombine(ui32 blockACrc, ui32 blockBCrc, size_t blockBSize) noexcept {
    return Singleton<TCrcUtilSse4>()->Combine(blockACrc, blockBCrc, blockBSize);
}

bool HaveFastCrc32c() noexcept {
    return TCrc::IsSSE42Available();
}
