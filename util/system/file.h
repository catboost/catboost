#pragma once

#include "fhandle.h"
#include "flock.h"

#include <util/generic/flags.h>
#include <util/generic/fwd.h>
#include <util/generic/ptr.h>
#include <util/generic/noncopyable.h>

#include <cstdio>

enum EOpenModeFlag {
    OpenExisting = 0,  // Opens a file. It fails if the file does not exist.
    TruncExisting = 1, // Opens a file and truncates it to zero size. It fails if the file does not exist.
    OpenAlways = 2,    // Opens a file, always. If a file does not exist, it creates a file.
    CreateNew = 3,     // Creates a new file. It fails if a specified file exists.
    CreateAlways = 4,  // Creates a new file, always. If a file exists, it overwrites the file.
    MaskCreation = 7,

    RdOnly = 8,  // open for reading only
    WrOnly = 16, // open for writing only
    RdWr = 24,   // open for reading and writing
    MaskRW = 24,

    Seq = 0x20,             // file access is primarily sequential (POSIX_FADV_SEQUENTIAL)
    Direct = 0x40,          // file is being opened with no system caching (Does not work as intended! See implementation)
    Temp = 0x80,            // avoid writing data back to disk if sufficient cache memory is available (no op for linux)
    ForAppend = 0x100,      // write appends data to the end of file (O_APPEND)
    Transient = 0x200,      // actually, temporary file - 'delete on close' for windows, unlink after creation for unix
    NoReuse = 0x400,        // no second access expected (POSIX_FADV_NOREUSE)
    CloseOnExec = 0x800,    // set close-on-exec right at open (O_CLOEXEC)
    DirectAligned = 0x1000, // file is actually being opened with no system caching (may require buffer alignment) (O_DIRECT)
    Sync = 0x2000,          // no write call will return before the data is transferred to the disk (O_SYNC)
    NoReadAhead = 0x4000,   // no sequential access expected, opposite for Seq (POSIX_FADV_RANDOM)

    AXOther = 0x00010000,
    AWOther = 0x00020000,
    AROther = 0x00040000,
    AXGroup = 0x00100000,
    AWGroup = 0x00200000,
    ARGroup = 0x00400000,
    AXUser = 0x01000000,
    AWUser = 0x02000000,
    ARUser = 0x04000000,
    AX = AXUser | AXGroup | AXOther,
    AW = AWUser | AWGroup,
    AR = ARUser | ARGroup | AROther,
    ARW = AR | AW,
    AMask = 0x0FFF0000,
};

Y_DECLARE_FLAGS(EOpenMode, EOpenModeFlag);
Y_DECLARE_OPERATORS_FOR_FLAGS(EOpenMode);

TString DecodeOpenMode(ui32 openMode);

enum SeekDir {
    sSet = 0,
    sCur = 1,
    sEnd = 2,
};

class TFileHandle: TMoveOnly {
public:
    constexpr TFileHandle() = default;

    /// Warning: takes ownership of fd, so closes it in destructor.
    inline TFileHandle(FHANDLE fd) noexcept
        : Fd_(fd)
    {
    }

    inline TFileHandle(TFileHandle&& other) noexcept
        : Fd_(other.Fd_)
    {
        other.Fd_ = INVALID_FHANDLE;
    }

    TFileHandle(const char* fName, EOpenMode oMode) noexcept;
    TFileHandle(const TString& fName, EOpenMode oMode) noexcept;
    TFileHandle(const std::filesystem::path& path, EOpenMode oMode) noexcept;

    inline ~TFileHandle() {
        Close();
    }

    TFileHandle& operator=(TFileHandle&& other) noexcept {
        Close();
        Fd_ = other.Release();
        return *this;
    }

    bool Close() noexcept;

    inline FHANDLE Release() noexcept {
        FHANDLE ret = Fd_;
        Fd_ = INVALID_FHANDLE;
        return ret;
    }

    inline void Swap(TFileHandle& r) noexcept {
        DoSwap(Fd_, r.Fd_);
    }

    inline operator FHANDLE() const noexcept {
        return Fd_;
    }

    inline bool IsOpen() const noexcept {
        return Fd_ != INVALID_FHANDLE;
    }

    i64 GetPosition() const noexcept;
    i64 GetLength() const noexcept;

    i64 Seek(i64 offset, SeekDir origin) noexcept;
    bool Resize(i64 length) noexcept;
    bool Reserve(i64 length) noexcept;
    bool FallocateNoResize(i64 length) noexcept;
    bool ShrinkToFit() noexcept;
    bool Flush() noexcept;
    //flush data only, without file metadata
    bool FlushData() noexcept;
    i32 Read(void* buffer, ui32 byteCount) noexcept;
    i32 Write(const void* buffer, ui32 byteCount) noexcept;
    i32 Pread(void* buffer, ui32 byteCount, i64 offset) const noexcept;
    i32 Pwrite(const void* buffer, ui32 byteCount, i64 offset) const noexcept;
    int Flock(int op) noexcept;

    FHANDLE Duplicate() const noexcept;
    int Duplicate2Posix(int dstHandle) const noexcept;

    //dup2 - like semantics, return true on success
    bool LinkTo(const TFileHandle& fh) const noexcept;

    //very low-level methods
    bool SetDirect();
    void ResetDirect();

    /* Manual file cache management, length = 0 means "as much as possible" */

    //measure amount of cached data in bytes, returns -1 if failed
    i64 CountCache(i64 offset = 0, i64 length = 0) const noexcept;
    //read data into cache and optionally wait for completion
    void PrefetchCache(i64 offset = 0, i64 length = 0, bool wait = true) const noexcept;
    //remove clean and unused data from cache
    void EvictCache(i64 offset = 0, i64 length = 0) const noexcept;
    //flush unwritten data in this range and optionally wait for completion
    bool FlushCache(i64 offset = 0, i64 length = 0, bool wait = true) noexcept;

private:
    FHANDLE Fd_ = INVALID_FHANDLE;
};

class TFile {
public:
    TFile();
    /// Takes ownership of handle, so closes it when the last holder of descriptor dies.
    explicit TFile(FHANDLE fd);
    TFile(FHANDLE fd, const TString& fname);
    TFile(const char* fName, EOpenMode oMode);
    TFile(const TString& fName, EOpenMode oMode);
    TFile(const std::filesystem::path& path, EOpenMode oMode);
    ~TFile();

    void Close();

    const TString& GetName() const noexcept;
    i64 GetPosition() const noexcept;
    i64 GetLength() const noexcept;
    bool IsOpen() const noexcept;
    FHANDLE GetHandle() const noexcept;

    i64 Seek(i64 offset, SeekDir origin);
    void Resize(i64 length);
    void Reserve(i64 length);
    void FallocateNoResize(i64 length);
    void ShrinkToFit();
    void Flush();
    void FlushData();

    void LinkTo(const TFile& f) const;
    TFile Duplicate() const;

    // Reads up to 1 GB without retrying, returns -1 on error
    i32 RawRead(void* buf, size_t len);
    // Reads up to 1 GB without retrying, throws on error
    size_t ReadOrFail(void* buf, size_t len);
    // Retries incomplete reads until EOF, throws on error
    size_t Read(void* buf, size_t len);
    // Reads exactly len bytes, throws on premature EOF or error
    void Load(void* buf, size_t len);

    // Retries incomplete writes, will either write len bytes or throw
    void Write(const void* buf, size_t len);

    // Retries incomplete reads until EOF, throws on error
    size_t Pread(void* buf, size_t len, i64 offset) const;
    // Single pread call
    i32 RawPread(void* buf, ui32 len, i64 offset) const;
    // Reads exactly len bytes, throws on premature EOF or error
    void Pload(void* buf, size_t len, i64 offset) const;

    // Retries incomplete writes, will either write len bytes or throw
    void Pwrite(const void* buf, size_t len, i64 offset) const;

    void Flock(int op);

    //do not use, their meaning very platform-dependant
    void SetDirect();
    void ResetDirect();

    /* Manual file cache management, length = 0 means "as much as possible" */

    //measure amount of cached data in bytes, returns -1 if failed
    i64 CountCache(i64 offset = 0, i64 length = 0) const noexcept;
    //read data into cache and optionally wait for completion
    void PrefetchCache(i64 offset = 0, i64 length = 0, bool wait = true) const noexcept;
    //remove clean and unused data from cache, incomplete pages could stay
    void EvictCache(i64 offset = 0, i64 length = 0) const noexcept;
    //flush unwritten data in this range and optionally wait for completion
    void FlushCache(i64 offset = 0, i64 length = 0, bool wait = true);

    static TFile Temporary(const TString& prefix);
    static TFile ForAppend(const TString& path);

private:
    class TImpl;
    TSimpleIntrusivePtr<TImpl> Impl_;
};

TFile Duplicate(FILE*);
TFile Duplicate(int);

bool PosixDisableReadAhead(FHANDLE fileHandle, void* addr) noexcept;
