#include "safe_memory_reader.h"

#ifdef _linux_
#include <fcntl.h>
#include <unistd.h>
#endif

#include <library/cpp/yt/assert/assert.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TSafeMemoryReader::TSafeMemoryReader()
{
#ifdef _linux_
    FD_ = ::open("/proc/self/mem", O_RDONLY);
    YT_VERIFY(FD_ >= 0);
#endif
}

TSafeMemoryReader::~TSafeMemoryReader()
{
#ifdef _linux_
    ::close(FD_);
#endif
}

bool TSafeMemoryReader::ReadRaw(const void* addr, void* ptr, size_t size)
{
#ifdef _linux_
    int ret;
    do {
        ret = ::pread64(FD_, ptr, size, reinterpret_cast<uintptr_t>(addr));
    } while (ret < 0 && errno == EINTR);
    return ret == static_cast<int>(size);
#else
    Y_UNUSED(FD_, addr, ptr, size);
    return false;
#endif
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
