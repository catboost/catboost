#pragma once

#include <util/generic/ptr.h>
#include <util/generic/guid.h>
#include <util/system/defaults.h>
#include <util/system/yassert.h>
#include <limits>

#if defined(_win32_) || defined(__IOS__)
#include <util/system/shmat.h>

struct TPosixSharedMemory: public TSharedMemory {
    using TSharedMemory::TSharedMemory;
    size_t GetSizeT() const {
        return (size_t)GetSize();
    }
};

#else
#if defined(_linux_)
#include <limits.h> // for PATH_MAX
#elif defined(_darwin_)
#include <sys/posix_shm.h>
#endif

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define INVALID_HANDLE_VALUE -1
#define INVALID_POINTER_VALUE NULL // user expects NULL as error result

// source-compatible with TSharedMemory (util/system/shmat.h)
class TPosixSharedMemory: public TThrRefBase {
public:
    enum EUnlinkPolicy { UP_DONT_UNLINK,
                         UP_CREATOR_UNLINKS,
                         UP_FORCE_UNLINK };

    TPosixSharedMemory()
        : Fd(INVALID_HANDLE_VALUE)
        , Ptr(INVALID_POINTER_VALUE)
        , Size(0)
        , IsCreator(false)
        , Unlinked(false)
    {
    }

    ~TPosixSharedMemory() override {
        Close(Unlinked ? UP_DONT_UNLINK : UP_CREATOR_UNLINKS);
    }

    bool Create(const size_t size, TGUID preferedGuid = TGUID()) {
        Y_ABORT_UNLESS(Guid.IsEmpty(), "You must call Close before");
        if (preferedGuid.IsEmpty()) {
            CreateGuid(&preferedGuid);
        }
        return CreateOpen(preferedGuid, size, true);
    }

    bool Open(const TGUID& guid, const size_t size) {
        Y_ABORT_UNLESS(Guid.IsEmpty(), "You must call Close before");
        return CreateOpen(guid, size, false);
    }

    // after unlinking Open will return false, and Create may map new region with same name.
    void Close(const EUnlinkPolicy up = UP_CREATOR_UNLINKS) // default argument for compatibility with TSharedMemory
    {
        Y_ASSERT((Ptr == nullptr) == (Size == 0));
        Y_ASSERT((Fd == INVALID_HANDLE_VALUE) == Guid.IsEmpty());

        if (Ptr != INVALID_POINTER_VALUE) {
            munmap(Ptr, Size);
        }
        Ptr = INVALID_POINTER_VALUE;
        Size = 0;

        if (Fd != INVALID_HANDLE_VALUE) {
            close(Fd);
            Unlink(up);
        }
        Fd = INVALID_HANDLE_VALUE;
        Guid = TGUID();

        IsCreator = false;
    }

    // Nobody can Open this shared memory again!
    // But this shared memory will be removed only when all attached processes exit (or terminate).
    // Careful! Calling this method simultaneously from multiple processes may lead to unlinking another's shared memory!
    // Better use with UP_CREATOR_UNLINKS flag.
    bool Unlink(const EUnlinkPolicy up) {
        // This method is the only reason to switch from System V shared memory (shmget/shmat/shmdt) to POSIX shared memory:
        // using System V shared memory after shmdt is unspecified behavior: smhdt decrements refcount and after its call
        // shared memory may be removed (if refcount == 0 && somebody called shmctl(IPC_RMID)) without waiting process to exit!
        // But without calling it right after attaching we may have "shared memory leak" after unexpected processes termination.
        // In the opposite unlinking POSIX shared memory is like unlinking files - nobody can open it again, but processes
        // which already opened it will see no difference.
        if (up == UP_FORCE_UNLINK || (IsCreator && up == UP_CREATOR_UNLINKS)) {
            Y_ASSERT(!Unlinked && !Guid.IsEmpty());
            Y_ABORT_UNLESS(!Unlinked, "You tried to unlink shared memory twice! Fix your code");
            Unlinked = shm_unlink(ConvertGuidToName(Guid).c_str());
            return Unlinked;
        }
        return true;
    }

    const TGUID& GetId() {
        return Guid;
    }
    void* GetPtr() {
        return Ptr;
    }
    int GetSize() const {
        return (int)Size;
    } // for compatibility with TSharedMemory
    size_t GetSizeT() const {
        return Size;
    }

private:
    static TString ConvertGuidToName(const TGUID& guid) {
        TString result;
        result.reserve(50);

        result += "/nl";
        result += GetGuidAsString(guid);

        size_t limit = std::numeric_limits<size_t>::max();
#if defined(_linux_)
        limit = PATH_MAX;
#elif defined(_freebsd_)
        limit = 1023;
#elif defined(_darwin_)
        limit = PSHMNAMLEN; // actually it's only 31 :-(
#endif
        if (result.size() + 1 > limit) { // +1 for null terminator
            result.erase(result.find_last_of('-'));
        }
        Y_ABORT_UNLESS(result.size() < limit, "Wow, your system really sucks!");

        return result;
    }

    bool CreateOpen(const TGUID& guid, const size_t size, const bool isCreate) {
        if (size > (size_t)std::numeric_limits<off_t>::max()) { // ftruncate will fail
            Y_DEBUG_ABORT_UNLESS(false, "size = %" PRIu64 " is too big for off_t", (ui64)size);
            errno = EFBIG;
            return false;
        }

        if (!CreateOpenImpl(guid, size, isCreate)) {
            const int e = errno;
            Close();
            errno = e;
            return false;
        }
        return true;
    }

    bool CreateOpenImpl(const TGUID& guid, const size_t size, const bool isCreate) {
        IsCreator = isCreate;

        int flags = O_RDWR;
        if (IsCreator) {
            flags |= O_CREAT | O_EXCL;
        }

        // sets Fd and Guid
        if (!ShmOpen(guid, flags)) {
            return false;
        }

        if (IsCreator && ftruncate(Fd, (off_t)size) < 0) {
            Y_DEBUG_ABORT_UNLESS(false, "errno = %d (%s)", errno, strerror(errno));
            return false;
        }

        // sets Ptr and Size
        if (!Mmap(size)) {
            return false;
        }

        return true;
    }

    bool ShmOpen(const TGUID& guid, const int flags) {
        Fd = shm_open(ConvertGuidToName(guid).c_str(), flags, 0666);
        if (Fd < 0) {
            Y_DEBUG_ABORT_UNLESS(false, "errno = %d (%s)", errno, strerror(errno));
            Fd = INVALID_HANDLE_VALUE;
            Guid = TGUID();
            return false;
        }
        Guid = guid;
        return true;
    }

    bool Mmap(const size_t size) {
        int flags = MAP_SHARED;
#ifdef _linux_
        // do not swap these pages
        //flags |= MAP_LOCKED;  // requires root, and may not fit everybody needs
#endif

        Ptr = mmap(nullptr, size, PROT_WRITE | PROT_READ, flags, Fd, 0);
        if (Ptr == MAP_FAILED) {
            Y_DEBUG_ABORT_UNLESS(false, "errno = %d (%s)", errno, strerror(errno));
            Ptr = INVALID_POINTER_VALUE;
            Size = 0;
            return false;
        }
        Size = size;
        return true;
    }

    TGUID Guid;
    int Fd;
    void* Ptr;
    size_t Size;
    bool IsCreator;
    bool Unlinked;
};

#endif
