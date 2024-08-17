#include "file.h"
#include "flock.h"
#include "fstat.h"
#include "sysstat.h"
#include "align.h"
#include "info.h"

#include <array>
#include <filesystem>

#include <util/string/util.h>
#include <util/string/cast.h>
#include <util/string/builder.h>

#include <util/stream/hex.h>
#include <util/stream/format.h>

#include <util/random/random.h>

#include <util/generic/size_literals.h>
#include <util/generic/string.h>
#include <util/generic/ylimits.h>
#include <util/generic/yexception.h>

#include <util/datetime/base.h>

#include <errno.h>

#if defined(_unix_)
    #include <fcntl.h>

    #if defined(_linux_) && (!defined(_android_) || __ANDROID_API__ >= 21) && !defined(FALLOC_FL_KEEP_SIZE)
        #include <linux/falloc.h>
    #endif

    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/mman.h>
#elif defined(_win_)
    #include "winint.h"
    #include "fs_win.h"
    #include <io.h>
#endif

#if defined(_bionic_)
    #include <sys/sendfile.h>
    #define HAVE_POSIX_FADVISE 0
    #define HAVE_SYNC_FILE_RANGE 0
#elif defined(_linux_)
    #include <sys/sendfile.h>
    #define HAVE_POSIX_FADVISE 1
    #define HAVE_SYNC_FILE_RANGE 1
#elif defined(__FreeBSD__) && !defined(WITH_VALGRIND)
    #include <sys/param.h>
    #define HAVE_POSIX_FADVISE (__FreeBSD_version >= 900501)
    #define HAVE_SYNC_FILE_RANGE 0
#else
    #define HAVE_POSIX_FADVISE 0
    #define HAVE_SYNC_FILE_RANGE 0
#endif

static bool IsStupidFlagCombination(EOpenMode oMode) {
    // ForAppend will actually not be applied in the following combinations:
    return (oMode & (CreateAlways | ForAppend)) == (CreateAlways | ForAppend) || (oMode & (TruncExisting | ForAppend)) == (TruncExisting | ForAppend) || (oMode & (CreateNew | ForAppend)) == (CreateNew | ForAppend);
}

#if defined(_win_)

static SECURITY_ATTRIBUTES ConvertToSecAttrs(EOpenMode oMode) {
    bool closeOnExec = (oMode & CloseOnExec);
    SECURITY_ATTRIBUTES secAttrs;
    secAttrs.bInheritHandle = closeOnExec ? FALSE : TRUE;
    secAttrs.lpSecurityDescriptor = nullptr;
    secAttrs.nLength = sizeof(secAttrs);
    return secAttrs;
}

TFileHandle::TFileHandle(const std::filesystem::path& path, EOpenMode oMode) noexcept {
    ui32 fcMode = 0;
    EOpenMode createMode = oMode & MaskCreation;
    Y_ABORT_UNLESS(!IsStupidFlagCombination(oMode), "oMode %d makes no sense", static_cast<int>(oMode));
    if (!(oMode & MaskRW)) {
        oMode |= RdWr;
    }
    if (!(oMode & AMask)) {
        oMode |= ARW;
    }

    switch (createMode) {
        case OpenExisting:
            fcMode = OPEN_EXISTING;
            break;
        case TruncExisting:
            fcMode = TRUNCATE_EXISTING;
            break;
        case OpenAlways:
            fcMode = OPEN_ALWAYS;
            break;
        case CreateNew:
            fcMode = CREATE_NEW;
            break;
        case CreateAlways:
            fcMode = CREATE_ALWAYS;
            break;
        default:
            abort();
            break;
    }

    ui32 faMode = 0;
    if (oMode & RdOnly) {
        faMode |= GENERIC_READ;
    }
    if (oMode & WrOnly) {
        // WrOnly or RdWr
        faMode |= GENERIC_WRITE;
    }
    if (oMode & ::ForAppend) {
        faMode |= GENERIC_WRITE;
        faMode |= FILE_APPEND_DATA;
        faMode &= ~FILE_WRITE_DATA;
    }

    ui32 shMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

    ui32 attrMode = FILE_ATTRIBUTE_NORMAL;
    if ((createMode == OpenExisting || createMode == OpenAlways) && ((oMode & AMask) == (oMode & AR))) {
        attrMode |= FILE_ATTRIBUTE_READONLY;
    }
    if (oMode & Seq) {
        attrMode |= FILE_FLAG_SEQUENTIAL_SCAN;
    }
    if (oMode & Temp) {
        // we use TTempFile instead of FILE_FLAG_DELETE_ON_CLOSE
        attrMode |= FILE_ATTRIBUTE_TEMPORARY;
    }
    if (oMode & Transient) {
        attrMode |= FILE_FLAG_DELETE_ON_CLOSE;
    }
    if ((oMode & (Direct | DirectAligned)) && (oMode & WrOnly)) {
        // WrOnly or RdWr
        attrMode |= /*FILE_FLAG_NO_BUFFERING |*/ FILE_FLAG_WRITE_THROUGH;
    }

    SECURITY_ATTRIBUTES secAttrs = ConvertToSecAttrs(oMode);

    Fd_ = ::CreateFileW(
        path.c_str(),
        faMode,
        shMode,
        &secAttrs,
        fcMode,
        attrMode,
        /* hTemplateHandle = */ nullptr);

    if ((oMode & ::ForAppend) && (Fd_ != INVALID_FHANDLE)) {
        ::SetFilePointer(Fd_, 0, 0, FILE_END);
    }
}

TFileHandle::TFileHandle(const TString& fName, EOpenMode oMode) noexcept
    : TFileHandle{
          // clang-format: off
          std::filesystem::path(
              std::u8string_view(reinterpret_cast<const char8_t*>(fName.data()), fName.size())),
          // clang-format: on
          oMode,
      }
{
}

#elif defined(_unix_)
TFileHandle::TFileHandle(const std::filesystem::path& path, EOpenMode oMode) noexcept {
    ui32 fcMode = 0;
    Y_ABORT_UNLESS(!IsStupidFlagCombination(oMode), "oMode %d makes no sense", static_cast<int>(oMode));
    if (!(oMode & MaskRW)) {
        oMode |= RdWr;
    }
    if (!(oMode & AMask)) {
        oMode |= ARW;
    }

    EOpenMode createMode = oMode & MaskCreation;
    switch (createMode) {
        case OpenExisting:
            fcMode = 0;
            break;
        case TruncExisting:
            fcMode = O_TRUNC;
            break;
        case OpenAlways:
            fcMode = O_CREAT;
            break;
        case CreateNew:
            fcMode = O_CREAT | O_EXCL;
            break;
        case CreateAlways:
            fcMode = O_CREAT | O_TRUNC;
            break;
        default:
            abort();
            break;
    }

    if ((oMode & RdOnly) && (oMode & WrOnly)) {
        fcMode |= O_RDWR;
    } else if (oMode & RdOnly) {
        fcMode |= O_RDONLY;
    } else if (oMode & WrOnly) {
        fcMode |= O_WRONLY;
    }

    if (oMode & ::ForAppend) {
        fcMode |= O_APPEND;
    }

    if (oMode & CloseOnExec) {
        fcMode |= O_CLOEXEC;
    }

    /* I don't now about this for unix...
    if (oMode & Temp) {
    }
    */
    #if defined(_freebsd_)
    if (oMode & (Direct | DirectAligned)) {
        fcMode |= O_DIRECT;
    }

    if (oMode & Sync) {
        fcMode |= O_SYNC;
    }
    #elif defined(_linux_)
    if (oMode & DirectAligned) {
        /*
         * O_DIRECT in Linux requires aligning request size and buffer address
         * to size of hardware sector (see hw_sector_size or ioctl BLKSSZGET).
         * Usually 512 bytes, but modern hardware works better with 4096 bytes.
         */
        fcMode |= O_DIRECT;
    }
    if (oMode & Sync) {
        fcMode |= O_SYNC;
    }
    #endif

    #if defined(_linux_)
    fcMode |= O_LARGEFILE;
    #endif

    ui32 permMode = 0;
    if (oMode & AXOther) {
        permMode |= S_IXOTH;
    }
    if (oMode & AWOther) {
        permMode |= S_IWOTH;
    }
    if (oMode & AROther) {
        permMode |= S_IROTH;
    }
    if (oMode & AXGroup) {
        permMode |= S_IXGRP;
    }
    if (oMode & AWGroup) {
        permMode |= S_IWGRP;
    }
    if (oMode & ARGroup) {
        permMode |= S_IRGRP;
    }
    if (oMode & AXUser) {
        permMode |= S_IXUSR;
    }
    if (oMode & AWUser) {
        permMode |= S_IWUSR;
    }
    if (oMode & ARUser) {
        permMode |= S_IRUSR;
    }

    do {
        Fd_ = ::open(path.c_str(), fcMode, permMode);
    } while (Fd_ == -1 && errno == EINTR);

    #if HAVE_POSIX_FADVISE
    if (Fd_ >= 0) {
        if (oMode & NoReuse) {
            ::posix_fadvise(Fd_, 0, 0, POSIX_FADV_NOREUSE);
        }

        if (oMode & Seq) {
            ::posix_fadvise(Fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
        }

        if (oMode & NoReadAhead) {
            ::posix_fadvise(Fd_, 0, 0, POSIX_FADV_RANDOM);
        }
    }
    #endif

    // temp file
    if (Fd_ >= 0 && (oMode & Transient)) {
        std::filesystem::remove(path);
    }
}

TFileHandle::TFileHandle(const TString& fName, EOpenMode oMode) noexcept
    : TFileHandle{
          std::filesystem::path(fName.ConstRef()),
          oMode,
      }
{
}
#else
    #error unsupported platform
#endif

TFileHandle::TFileHandle(const char* fName, EOpenMode oMode) noexcept
    : TFileHandle(TString(fName), oMode)
{
}

bool TFileHandle::Close() noexcept {
    bool isOk = true;
#ifdef _win_
    if (Fd_ != INVALID_FHANDLE) {
        isOk = (::CloseHandle(Fd_) != 0);
    }
    if (!isOk) {
        Y_ABORT_UNLESS(GetLastError() != ERROR_INVALID_HANDLE,
                       "must not quietly close invalid handle");
    }
#elif defined(_unix_)
    if (Fd_ != INVALID_FHANDLE) {
        isOk = (::close(Fd_) == 0 || errno == EINTR);
    }
    if (!isOk) {
        // Do not quietly close bad descriptor,
        // because often it means double close
        // that is disasterous
        Y_ABORT_UNLESS(errno != EBADF, "must not quietly close bad descriptor: fd=%d", int(Fd_));
    }
#else
    #error unsupported platform
#endif
    Fd_ = INVALID_FHANDLE;
    return isOk;
}

static inline i64 DoSeek(FHANDLE h, i64 offset, SeekDir origin) noexcept {
    if (h == INVALID_FHANDLE) {
        return -1L;
    }
#if defined(_win_)
    static ui32 dir[] = {FILE_BEGIN, FILE_CURRENT, FILE_END};
    LARGE_INTEGER pos;
    pos.QuadPart = offset;
    pos.LowPart = ::SetFilePointer(h, pos.LowPart, &pos.HighPart, dir[origin]);
    if (pos.LowPart == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) {
        pos.QuadPart = -1;
    }
    return pos.QuadPart;
#elif defined(_unix_)
    static int dir[] = {SEEK_SET, SEEK_CUR, SEEK_END};
    #if defined(_sun_)
    return ::llseek(h, (offset_t)offset, dir[origin]);
    #else
    return ::lseek(h, (off_t)offset, dir[origin]);
    #endif
#else
    #error unsupported platform
#endif
}

i64 TFileHandle::GetPosition() const noexcept {
    return DoSeek(Fd_, 0, sCur);
}

i64 TFileHandle::Seek(i64 offset, SeekDir origin) noexcept {
    return DoSeek(Fd_, offset, origin);
}

i64 TFileHandle::GetLength() const noexcept {
    // XXX: returns error code, but does not set errno
    if (!IsOpen()) {
        return -1L;
    }
    return GetFileLength(Fd_);
}

bool TFileHandle::Resize(i64 length) noexcept {
    if (!IsOpen()) {
        return false;
    }
    i64 currentLength = GetLength();
    if (length == currentLength) {
        return true;
    }
#if defined(_win_)
    i64 currentPosition = GetPosition();
    if (currentPosition == -1L) {
        return false;
    }
    Seek(length, sSet);
    if (!::SetEndOfFile(Fd_)) {
        return false;
    }
    if (currentPosition < length) {
        Seek(currentPosition, sSet);
    }
    return true;
#elif defined(_unix_)
    return (0 == ftruncate(Fd_, (off_t)length));
#else
    #error unsupported platform
#endif
}

bool TFileHandle::Reserve(i64 length) noexcept {
    // FIXME this should reserve disk space with fallocate
    if (!IsOpen()) {
        return false;
    }
    i64 currentLength = GetLength();
    if (length <= currentLength) {
        return true;
    }
    if (!Resize(length)) {
        return false;
    }
#if defined(_win_)
    if (!::SetFileValidData(Fd_, length)) {
        Resize(currentLength);
        return false;
    }
#elif defined(_unix_)
// No way to implement this under FreeBSD. Just do nothing
#else
    #error unsupported platform
#endif
    return true;
}

bool TFileHandle::FallocateNoResize(i64 length) noexcept {
    if (!IsOpen()) {
        return false;
    }
#if defined(_linux_) && (!defined(_android_) || __ANDROID_API__ >= 21)
    return !fallocate(Fd_, FALLOC_FL_KEEP_SIZE, 0, length);
#elif defined(_win_)
    FILE_ALLOCATION_INFO allocInfo = {};
    allocInfo.AllocationSize.QuadPart = length;

    return SetFileInformationByHandle(Fd_, FileAllocationInfo, &allocInfo,
                                      sizeof(FILE_ALLOCATION_INFO));
#else
    Y_UNUSED(length);
    return true;
#endif
}

// Pair for FallocateNoResize
bool TFileHandle::ShrinkToFit() noexcept {
    if (!IsOpen()) {
        return false;
    }
#if defined(_linux_) && (!defined(_android_) || __ANDROID_API__ >= 21)
    return !ftruncate(Fd_, (off_t)GetLength());
#else
    return true;
#endif
}

bool TFileHandle::Flush() noexcept {
    if (!IsOpen()) {
        return false;
    }
#if defined(_win_)
    bool ok = ::FlushFileBuffers(Fd_) != 0;
    /*
     * FlushFileBuffers fails if hFile is a handle to the console output.
     * That is because the console output is not buffered.
     * The function returns FALSE, and GetLastError returns ERROR_INVALID_HANDLE.
     */
    return ok || GetLastError() == ERROR_INVALID_HANDLE;
#elif defined(_unix_)
    int ret = ::fsync(Fd_);

    /*
     * Ignore EROFS, EINVAL - fd is bound to a special file
     * (PIPE, FIFO, or socket) which does not support synchronization.
     * Fail in case of EIO, ENOSPC, EDQUOT - data might be lost.
     */
    return ret == 0 || errno == EROFS || errno == EINVAL
    #if defined(_darwin_)
           // ENOTSUP fd does not refer to a vnode
           || errno == ENOTSUP
    #endif
        ;
#else
    #error unsupported platform
#endif
}

bool TFileHandle::FlushData() noexcept {
#if defined(_linux_)
    if (!IsOpen()) {
        return false;
    }

    int ret = ::fdatasync(Fd_);

    // Same loginc in error handling as for fsync above.
    return ret == 0 || errno == EROFS || errno == EINVAL;
#else
    return Flush();
#endif
}

i32 TFileHandle::Read(void* buffer, ui32 byteCount) noexcept {
    // FIXME size and return must be 64-bit
    if (!IsOpen()) {
        return -1;
    }
#if defined(_win_)
    DWORD bytesRead = 0;
    if (::ReadFile(Fd_, buffer, byteCount, &bytesRead, nullptr)) {
        return bytesRead;
    }
    return -1;
#elif defined(_unix_)
    i32 ret;
    do {
        ret = ::read(Fd_, buffer, byteCount);
    } while (ret == -1 && errno == EINTR);
    return ret;
#else
    #error unsupported platform
#endif
}

i32 TFileHandle::Write(const void* buffer, ui32 byteCount) noexcept {
    if (!IsOpen()) {
        return -1;
    }
#if defined(_win_)
    DWORD bytesWritten = 0;
    if (::WriteFile(Fd_, buffer, byteCount, &bytesWritten, nullptr)) {
        return bytesWritten;
    }
    return -1;
#elif defined(_unix_)
    i32 ret;
    do {
        ret = ::write(Fd_, buffer, byteCount);
    } while (ret == -1 && errno == EINTR);
    return ret;
#else
    #error unsupported platform
#endif
}

i32 TFileHandle::Pread(void* buffer, ui32 byteCount, i64 offset) const noexcept {
#if defined(_win_)
    OVERLAPPED io;
    Zero(io);
    DWORD bytesRead = 0;
    io.Offset = (ui32)offset;
    io.OffsetHigh = (ui32)(offset >> 32);
    if (::ReadFile(Fd_, buffer, byteCount, &bytesRead, &io)) {
        return bytesRead;
    }
    if (::GetLastError() == ERROR_HANDLE_EOF) {
        return 0;
    }
    return -1;
#elif defined(_unix_)
    i32 ret;
    do {
        ret = ::pread(Fd_, buffer, byteCount, offset);
    } while (ret == -1 && errno == EINTR);
    return ret;
#else
    #error unsupported platform
#endif
}

i32 TFileHandle::Pwrite(const void* buffer, ui32 byteCount, i64 offset) const noexcept {
#if defined(_win_)
    OVERLAPPED io;
    Zero(io);
    DWORD bytesWritten = 0;
    io.Offset = (ui32)offset;
    io.OffsetHigh = (ui32)(offset >> 32);
    if (::WriteFile(Fd_, buffer, byteCount, &bytesWritten, &io)) {
        return bytesWritten;
    }
    return -1;
#elif defined(_unix_)
    i32 ret;
    do {
        ret = ::pwrite(Fd_, buffer, byteCount, offset);
    } while (ret == -1 && errno == EINTR);
    return ret;
#else
    #error unsupported platform
#endif
}

FHANDLE TFileHandle::Duplicate() const noexcept {
    if (!IsOpen()) {
        return INVALID_FHANDLE;
    }
#if defined(_win_)
    FHANDLE dupHandle;
    if (!::DuplicateHandle(GetCurrentProcess(), Fd_, GetCurrentProcess(), &dupHandle, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
        return INVALID_FHANDLE;
    }
    return dupHandle;
#elif defined(_unix_)
    return ::dup(Fd_);
#else
    #error unsupported platform
#endif
}

int TFileHandle::Duplicate2Posix(int dstHandle) const noexcept {
    if (!IsOpen()) {
        return -1;
    }
#if defined(_win_)
    FHANDLE dupHandle = Duplicate();
    if (dupHandle == INVALID_FHANDLE) {
        _set_errno(EMFILE);
        return -1;
    }
    int posixHandle = _open_osfhandle((intptr_t)dupHandle, 0);
    if (posixHandle == -1) {
        CloseHandle(dupHandle);
        return -1;
    }
    if (dup2(posixHandle, dstHandle) == -1) {
        dstHandle = -1;
    }
    _close(posixHandle);
    return dstHandle;
#elif defined(_unix_)
    while (dup2(Fd_, dstHandle) == -1) {
        if (errno != EINTR) {
            return -1;
        }
    }
    return dstHandle;
#else
    #error unsupported platform
#endif
}

bool TFileHandle::LinkTo(const TFileHandle& fh) const noexcept {
#if defined(_unix_)
    while (dup2(fh.Fd_, Fd_) == -1) {
        if (errno != EINTR) {
            return false;
        }
    }
    return true;
#elif defined(_win_)
    TFileHandle nh(fh.Duplicate());

    if (!nh.IsOpen()) {
        return false;
    }

    // not thread-safe
    nh.Swap(*const_cast<TFileHandle*>(this));

    return true;
#else
    #error unsupported
#endif
}

int TFileHandle::Flock(int op) noexcept {
    return ::Flock(Fd_, op);
}

bool TFileHandle::SetDirect() {
#ifdef _linux_
    const long flags = fcntl(Fd_, F_GETFL);
    const int r = fcntl(Fd_, F_SETFL, flags | O_DIRECT);

    return !r;
#endif

    return false;
}

void TFileHandle::ResetDirect() {
#ifdef _linux_
    long flags = fcntl(Fd_, F_GETFL);
    fcntl(Fd_, F_SETFL, flags & ~O_DIRECT);
#endif
}

i64 TFileHandle::CountCache(i64 offset, i64 length) const noexcept {
#ifdef _linux_
    const i64 pageSize = NSystemInfo::GetPageSize();
    constexpr size_t vecSize = 512; // Fetch up to 2MiB at once
    const i64 batchSize = vecSize * pageSize;
    std::array<ui8, vecSize> vec;
    void* ptr = nullptr;
    i64 res = 0;

    if (!IsOpen()) {
        return -1;
    }

    if (!length) {
        length = GetLength();
        length -= Min(length, offset);
    }

    if (!length) {
        return 0;
    }

    const i64 begin = AlignDown(offset, pageSize);
    const i64 end = AlignUp(offset + length, pageSize);
    const i64 size = end - begin;

    /*
     * Since fincode is not implemented yet use mmap and mincore.
     * This is not so effective and scalable for frequent usage.
     */
    ptr = ::mmap(
        (caddr_t) nullptr,
        size,
        PROT_READ,
        MAP_SHARED | MAP_NORESERVE,
        Fd_,
        begin);
    if (MAP_FAILED == ptr) {
        return -1;
    }

    for (i64 base = begin; base < end; base += batchSize) {
        const size_t batch = Min(vecSize, size_t((end - base) / pageSize));
        void* batchPtr = static_cast<caddr_t>(ptr) + (base - begin);

        if (::mincore(batchPtr, batch * pageSize, vec.data())) {
            res = -1;
            break;
        }

        for (size_t i = 0; i < batch; i++) {
            // count uptodate complete pages in cache
            if (vec[i] & 1) {
                res += pageSize;
            }
        }

        if (base == begin && (vec[0] & 1)) {
            // cut head of first page
            res -= offset - begin;
        }

        if ((end - base) <= batchSize && (vec[batch - 1] & 1)) {
            // cut tail of last page
            res -= size - (offset - begin) - length;
        }
    }

    ::munmap(ptr, size);

    return res;
#else
    Y_UNUSED(offset);
    Y_UNUSED(length);
    return -1;
#endif
}

void TFileHandle::PrefetchCache(i64 offset, i64 length, bool wait) const noexcept {
#ifdef _linux_
    #if HAVE_POSIX_FADVISE
    // POSIX_FADV_WILLNEED starts reading upto read_ahead_kb in background
    ::posix_fadvise(Fd_, offset, length, POSIX_FADV_WILLNEED);
    #endif

    if (wait) {
        TFileHandle devnull("/dev/null", OpenExisting | WrOnly | CloseOnExec);
        off_t end = length ? (offset + length) : GetLength();
        off_t pos = offset;
        ssize_t ret;

        do {
            ret = ::sendfile((FHANDLE)devnull, Fd_, &pos, end - pos);
        } while (pos < end && (ret > 0 || errno == EINTR));
    }
#else
    Y_UNUSED(offset);
    Y_UNUSED(length);
    Y_UNUSED(wait);
#endif
}

void TFileHandle::EvictCache(i64 offset, i64 length) const noexcept {
#if HAVE_POSIX_FADVISE
    /*
     * This tries to evicts only unmaped, clean, complete pages.
     */
    ::posix_fadvise(Fd_, offset, length, POSIX_FADV_DONTNEED);
#else
    Y_UNUSED(offset);
    Y_UNUSED(length);
#endif
}

bool TFileHandle::FlushCache(i64 offset, i64 length, bool wait) noexcept {
#if HAVE_SYNC_FILE_RANGE
    int flags = SYNC_FILE_RANGE_WRITE;
    if (wait) {
        flags |= SYNC_FILE_RANGE_WAIT_AFTER;
    }
    int ret = ::sync_file_range(Fd_, offset, length, flags);
    return ret == 0 || errno == EROFS;
#else
    Y_UNUSED(offset);
    Y_UNUSED(length);
    if (wait) {
        return FlushData();
    }
    return true;
#endif
}

TString DecodeOpenMode(ui32 mode0) {
    ui32 mode = mode0;

    TStringBuilder r;

    struct TFlagCombo {
        ui32 Value;
        TStringBuf Name;
    };

    static constexpr TFlagCombo knownFlagCombos[]{

#define F(flag) {flag, #flag}

        F(RdWr),
        F(RdOnly),
        F(WrOnly),

        F(CreateAlways),
        F(CreateNew),
        F(OpenAlways),
        F(TruncExisting),
        F(ForAppend),
        F(Transient),
        F(CloseOnExec),

        F(Temp),
        F(Sync),
        F(Direct),
        F(DirectAligned),
        F(Seq),
        F(NoReuse),
        F(NoReadAhead),

        F(AX),
        F(AR),
        F(AW),
        F(ARW),

        F(AXOther),
        F(AWOther),
        F(AROther),
        F(AXGroup),
        F(AWGroup),
        F(ARGroup),
        F(AXUser),
        F(AWUser),
        F(ARUser),

#undef F

    };

    for (const auto& [flag, name] : knownFlagCombos) {
        if ((mode & flag) == flag) {
            mode &= ~flag;
            if (r) {
                r << '|';
            }
            r << name;
        }
    }

    if (mode != 0) {
        if (r) {
            r << TStringBuf("|");
        }

        r << Hex(mode);
    }

    if (!r) {
        return "0";
    }

    return std::move(r);
}

class TFile::TImpl: public TAtomicRefCount<TImpl> {
public:
    inline TImpl(FHANDLE fd, const TString& fname = TString())
        : Handle_(fd)
        , FileName_(fname)
    {
    }

    inline TImpl(const char* fName, EOpenMode oMode)
        : Handle_(fName, oMode)
        , FileName_(fName)
    {
        if (!Handle_.IsOpen()) {
            ythrow TFileError() << "can't open " << FileName_.Quote() << " with mode " << DecodeOpenMode(oMode) << " (" << Hex(oMode.ToBaseType()) << ")";
        }
    }

    inline TImpl(const TString& fName, EOpenMode oMode)
        : Handle_(fName, oMode)
        , FileName_(fName)
    {
        if (!Handle_.IsOpen()) {
            ythrow TFileError() << "can't open " << FileName_.Quote() << " with mode " << DecodeOpenMode(oMode) << " (" << Hex(oMode.ToBaseType()) << ")";
        }
    }

    inline TImpl(const std::filesystem::path& path, EOpenMode oMode)
        : Handle_(path, oMode)
        , FileName_(path.string())
    {
        if (!Handle_.IsOpen()) {
            ythrow TFileError() << "can't open " << FileName_.Quote() << " with mode " << DecodeOpenMode(oMode) << " (" << Hex(oMode.ToBaseType()) << ")";
        }
    }

    inline ~TImpl() = default;

    inline void Close() {
        if (!Handle_.Close()) {
            ythrow TFileError() << "can't close " << FileName_.Quote();
        }
    }

    const TString& GetName() const noexcept {
        return FileName_;
    }

    void SetName(const TString& newName) {
        FileName_ = newName;
    }

    const TFileHandle& GetHandle() const noexcept {
        return Handle_;
    }

    i64 Seek(i64 offset, SeekDir origin) {
        i64 pos = Handle_.Seek(offset, origin);
        if (pos == -1L) {
            ythrow TFileError() << "can't seek " << offset << " bytes in " << FileName_.Quote();
        }
        return pos;
    }

    void Resize(i64 length) {
        if (!Handle_.Resize(length)) {
            ythrow TFileError() << "can't resize " << FileName_.Quote() << " to size " << length;
        }
    }

    void Reserve(i64 length) {
        if (!Handle_.Reserve(length)) {
            ythrow TFileError() << "can't reserve " << length << " for file " << FileName_.Quote();
        }
    }

    void FallocateNoResize(i64 length) {
        if (!Handle_.FallocateNoResize(length)) {
            ythrow TFileError() << "can't allocate " << length << "bytes of space for file " << FileName_.Quote();
        }
    }

    void ShrinkToFit() {
        if (!Handle_.ShrinkToFit()) {
            ythrow TFileError() << "can't shrink " << FileName_.Quote() << " to logical size";
        }
    }

    void Flush() {
        if (!Handle_.Flush()) {
            ythrow TFileError() << "can't flush " << FileName_.Quote();
        }
    }

    void FlushData() {
        if (!Handle_.FlushData()) {
            ythrow TFileError() << "can't flush data " << FileName_.Quote();
        }
    }

    TFile Duplicate() const {
        TFileHandle dupH(Handle_.Duplicate());
        if (!dupH.IsOpen()) {
            ythrow TFileError() << "can't duplicate the handle of " << FileName_.Quote();
        }
        TFile res(dupH);
        dupH.Release();
        return res;
    }

    // Maximum amount of bytes to be read via single system call.
    // Some libraries fail when it is greater than max int.
    // Syscalls can cause contention if they operate on very large data blocks.
    static constexpr size_t MaxReadPortion = 1_GB;

    i32 RawRead(void* bufferIn, size_t numBytes) {
        const size_t toRead = Min(MaxReadPortion, numBytes);
        return Handle_.Read(bufferIn, toRead);
    }

    size_t ReadOrFail(void* buf, size_t numBytes) {
        const i32 reallyRead = RawRead(buf, numBytes);

        if (reallyRead < 0) {
            ythrow TFileError() << "can not read data from " << FileName_.Quote();
        }

        return reallyRead;
    }

    size_t Read(void* bufferIn, size_t numBytes) {
        ui8* buf = (ui8*)bufferIn;

        while (numBytes) {
            const size_t reallyRead = ReadOrFail(buf, numBytes);

            if (reallyRead == 0) {
                // file exhausted
                break;
            }

            buf += reallyRead;
            numBytes -= reallyRead;
        }

        return buf - (ui8*)bufferIn;
    }

    void Load(void* buf, size_t len) {
        if (Read(buf, len) != len) {
            ythrow TFileError() << "can't read " << len << " bytes from " << FileName_.Quote();
        }
    }

    // Maximum amount of bytes to be written via single system call.
    // Some libraries fail when it is greater than max int.
    // Syscalls can cause contention if they operate on very large data blocks.
    static constexpr size_t MaxWritePortion = 1_GB;

    void Write(const void* buffer, size_t numBytes) {
        const ui8* buf = (const ui8*)buffer;

        while (numBytes) {
            const i32 toWrite = (i32)Min(MaxWritePortion, numBytes);
            const i32 reallyWritten = Handle_.Write(buf, toWrite);

            if (reallyWritten < 0) {
                ythrow TFileError() << "can't write " << toWrite << " bytes to " << FileName_.Quote();
            }

            buf += reallyWritten;
            numBytes -= reallyWritten;
        }
    }

    size_t Pread(void* bufferIn, size_t numBytes, i64 offset) const {
        ui8* buf = (ui8*)bufferIn;

        while (numBytes) {
            const i32 toRead = (i32)Min(MaxReadPortion, numBytes);
            const i32 reallyRead = RawPread(buf, toRead, offset);

            if (reallyRead < 0) {
                ythrow TFileError() << "can not read data from " << FileName_.Quote();
            }

            if (reallyRead == 0) {
                // file exausted
                break;
            }

            buf += reallyRead;
            offset += reallyRead;
            numBytes -= reallyRead;
        }

        return buf - (ui8*)bufferIn;
    }

    i32 RawPread(void* buf, ui32 len, i64 offset) const {
        return Handle_.Pread(buf, len, offset);
    }

    void Pload(void* buf, size_t len, i64 offset) const {
        if (Pread(buf, len, offset) != len) {
            ythrow TFileError() << "can't read " << len << " bytes at offset " << offset << " from " << FileName_.Quote();
        }
    }

    void Pwrite(const void* buffer, size_t numBytes, i64 offset) const {
        const ui8* buf = (const ui8*)buffer;

        while (numBytes) {
            const i32 toWrite = (i32)Min(MaxWritePortion, numBytes);
            const i32 reallyWritten = Handle_.Pwrite(buf, toWrite, offset);

            if (reallyWritten < 0) {
                ythrow TFileError() << "can't write " << toWrite << " bytes to " << FileName_.Quote();
            }

            buf += reallyWritten;
            offset += reallyWritten;
            numBytes -= reallyWritten;
        }
    }

    void Flock(int op) {
        if (0 != Handle_.Flock(op)) {
            ythrow TFileError() << "can't flock " << FileName_.Quote();
        }
    }

    void SetDirect() {
        if (!Handle_.SetDirect()) {
            ythrow TFileError() << "can't set direct mode for " << FileName_.Quote();
        }
    }

    void ResetDirect() {
        Handle_.ResetDirect();
    }

    i64 CountCache(i64 offset, i64 length) const noexcept {
        return Handle_.CountCache(offset, length);
    }

    void PrefetchCache(i64 offset, i64 length, bool wait) const noexcept {
        Handle_.PrefetchCache(offset, length, wait);
    }

    void EvictCache(i64 offset, i64 length) const noexcept {
        Handle_.EvictCache(offset, length);
    }

    void FlushCache(i64 offset, i64 length, bool wait) {
        if (!Handle_.FlushCache(offset, length, wait)) {
            ythrow TFileError() << "can't flush data " << FileName_.Quote();
        }
    }

private:
    TFileHandle Handle_;
    TString FileName_;
};

TFile::TFile()
    : Impl_(new TImpl(INVALID_FHANDLE))
{
}

TFile::TFile(FHANDLE fd)
    : Impl_(new TImpl(fd))
{
}

TFile::TFile(FHANDLE fd, const TString& name)
    : Impl_(new TImpl(fd, name))
{
}

TFile::TFile(const char* fName, EOpenMode oMode)
    : Impl_(new TImpl(fName, oMode))
{
}

TFile::TFile(const TString& fName, EOpenMode oMode)
    : Impl_(new TImpl(fName, oMode))
{
}

TFile::TFile(const std::filesystem::path& path, EOpenMode oMode)
    : Impl_(new TImpl(path, oMode))
{
}

TFile::~TFile() = default;

void TFile::Close() {
    Impl_->Close();
}

const TString& TFile::GetName() const noexcept {
    return Impl_->GetName();
}

i64 TFile::GetPosition() const noexcept {
    return Impl_->GetHandle().GetPosition();
}

i64 TFile::GetLength() const noexcept {
    return Impl_->GetHandle().GetLength();
}

bool TFile::IsOpen() const noexcept {
    return Impl_->GetHandle().IsOpen();
}

FHANDLE TFile::GetHandle() const noexcept {
    return Impl_->GetHandle();
}

i64 TFile::Seek(i64 offset, SeekDir origin) {
    return Impl_->Seek(offset, origin);
}

void TFile::Resize(i64 length) {
    Impl_->Resize(length);
}

void TFile::Reserve(i64 length) {
    Impl_->Reserve(length);
}

void TFile::FallocateNoResize(i64 length) {
    Impl_->FallocateNoResize(length);
}

void TFile::ShrinkToFit() {
    Impl_->ShrinkToFit();
}

void TFile::Flush() {
    Impl_->Flush();
}

void TFile::FlushData() {
    Impl_->FlushData();
}

TFile TFile::Duplicate() const {
    TFile res = Impl_->Duplicate();
    res.Impl_->SetName(Impl_->GetName());
    return res;
}

size_t TFile::Read(void* buf, size_t len) {
    return Impl_->Read(buf, len);
}

i32 TFile::RawRead(void* buf, size_t len) {
    return Impl_->RawRead(buf, len);
}

size_t TFile::ReadOrFail(void* buf, size_t len) {
    return Impl_->ReadOrFail(buf, len);
}

void TFile::Load(void* buf, size_t len) {
    Impl_->Load(buf, len);
}

void TFile::Write(const void* buf, size_t len) {
    Impl_->Write(buf, len);
}

size_t TFile::Pread(void* buf, size_t len, i64 offset) const {
    return Impl_->Pread(buf, len, offset);
}

i32 TFile::RawPread(void* buf, ui32 len, i64 offset) const {
    return Impl_->RawPread(buf, len, offset);
}

void TFile::Pload(void* buf, size_t len, i64 offset) const {
    Impl_->Pload(buf, len, offset);
}

void TFile::Pwrite(const void* buf, size_t len, i64 offset) const {
    Impl_->Pwrite(buf, len, offset);
}

void TFile::Flock(int op) {
    Impl_->Flock(op);
}

void TFile::SetDirect() {
    Impl_->SetDirect();
}

void TFile::ResetDirect() {
    Impl_->ResetDirect();
}

i64 TFile::CountCache(i64 offset, i64 length) const noexcept {
    return Impl_->CountCache(offset, length);
}

void TFile::PrefetchCache(i64 offset, i64 length, bool wait) const noexcept {
    Impl_->PrefetchCache(offset, length, wait);
}

void TFile::EvictCache(i64 offset, i64 length) const noexcept {
    Impl_->EvictCache(offset, length);
}

void TFile::FlushCache(i64 offset, i64 length, bool wait) {
    Impl_->FlushCache(offset, length, wait);
}

void TFile::LinkTo(const TFile& f) const {
    if (!Impl_->GetHandle().LinkTo(f.Impl_->GetHandle())) {
        ythrow TFileError() << "can not link fd(" << GetName() << " -> " << f.GetName() << ")";
    }
}

TFile TFile::Temporary(const TString& prefix) {
    // TODO - handle impossible case of name collision
    return TFile(prefix + ToString(MicroSeconds()) + "-" + ToString(RandomNumber<ui64>()), CreateNew | RdWr | Seq | Temp | Transient);
}

TFile TFile::ForAppend(const TString& path) {
    return TFile(path, OpenAlways | WrOnly | Seq | ::ForAppend);
}

TFile Duplicate(FILE* f) {
    return Duplicate(fileno(f));
}

TFile Duplicate(int fd) {
#if defined(_win_)
    /* There are two options of how to duplicate a file descriptor on Windows:
     *
     * 1:
     * - Call dup.
     * - Call _get_osfhandle on the result.
     * - Use returned handle.
     * - Call _close on file descriptor returned by dup. This will also close
     *   the handle.
     *
     * 2:
     * - Call _get_osfhandle.
     * - Call DuplicateHandle on the result.
     * - Use returned handle.
     * - Call CloseHandle.
     *
     * TFileHandle calls CloseHandle when destroyed, leaving us with option #2. */
    FHANDLE handle = reinterpret_cast<FHANDLE>(::_get_osfhandle(fd));

    FHANDLE dupHandle;
    if (!::DuplicateHandle(GetCurrentProcess(), handle, GetCurrentProcess(), &dupHandle, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
        ythrow TFileError() << "can not duplicate file descriptor " << LastSystemError() << Endl;
    }

    return TFile(dupHandle);
#elif defined(_unix_)
    return TFile(::dup(fd));
#else
    #error unsupported platform
#endif
}

bool PosixDisableReadAhead(FHANDLE fileHandle, void* addr) noexcept {
    int ret = -1;

#if HAVE_POSIX_FADVISE
    #if defined(_linux_)
    Y_UNUSED(fileHandle);
    ret = madvise(addr, 0, MADV_RANDOM); // according to klamm@ posix_fadvise does not work under linux, madvise does work
    #else
    Y_UNUSED(addr);
    ret = ::posix_fadvise(fileHandle, 0, 0, POSIX_FADV_RANDOM);
    #endif
#else
    Y_UNUSED(fileHandle);
    Y_UNUSED(addr);
#endif
    return ret == 0;
}
