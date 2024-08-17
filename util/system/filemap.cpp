#include "info.h"
#include "madvise.h"
#include "defaults.h"
#include "hi_lo.h"

#include <util/generic/yexception.h>
#include <util/generic/singleton.h>

#if defined(_win_)
    #include "winint.h"
#elif defined(_unix_)
    #include <sys/types.h>
    #include <sys/mman.h>

    #if !defined(_linux_) && !defined(_emscripten_)
        #ifdef MAP_POPULATE
            #error unlisted platform supporting MAP_POPULATE
        #endif
        #define MAP_POPULATE 0
    #endif

    #if !defined(_freebsd_)
        #ifdef MAP_NOCORE
            #error unlisted platform supporting MAP_NOCORE
        #endif
        #define MAP_NOCORE 0
    #endif
#else
    #error todo
#endif

#include <util/generic/utility.h>
#include <util/system/sanitizers.h>
#include "filemap.h"

#undef PAGE_SIZE
#undef GRANULARITY

#ifdef _win_
    #define MAP_FAILED ((void*)(LONG_PTR)-1)
#endif

namespace {
    struct TSysInfo {
        inline TSysInfo()
            : GRANULARITY_(CalcGranularity())
            , PAGE_SIZE_(NSystemInfo::GetPageSize())
        {
        }

        static inline const TSysInfo& Instance() {
            return *Singleton<TSysInfo>();
        }

        static inline size_t CalcGranularity() noexcept {
#if defined(_win_)
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            return sysInfo.dwAllocationGranularity;
#else
            return NSystemInfo::GetPageSize();
#endif
        }

        const size_t GRANULARITY_;
        const size_t PAGE_SIZE_;
    };
}

#define GRANULARITY (TSysInfo::Instance().GRANULARITY_)
#define PAGE_SIZE (TSysInfo::Instance().PAGE_SIZE_)

const TString& TMemoryMapCommon::UnknownFileName() {
    static const TString unknownFileName = "Unknown_file_name";
    return unknownFileName;
}

static inline i64 DownToGranularity(i64 offset) noexcept {
    return offset & ~((i64)(GRANULARITY - 1));
}

#if defined(_unix_)
static int ModeToMmapFlags(TMemoryMapCommon::EOpenMode mode) {
    int flags = MAP_NOCORE;
    if ((mode & TMemoryMapCommon::oAccessMask) == TMemoryMapCommon::oCopyOnWr) {
        flags |= MAP_PRIVATE;
    } else {
        flags |= MAP_SHARED;
    }
    if (mode & TMemoryMapCommon::oPopulate) {
        flags |= MAP_POPULATE;
    }
    return flags;
}

static int ModeToMmapProt(TMemoryMapCommon::EOpenMode mode) {
    int prot = PROT_READ;
    if ((mode & TMemoryMapCommon::oAccessMask) != TMemoryMapCommon::oRdOnly) {
        prot |= PROT_WRITE;
    }
    return prot;
}
#endif

// maybe we should move this function to another .cpp file to avoid unwanted optimization?
void NPrivate::Precharge(const void* data, size_t dataSize, size_t off, size_t size) {
    if (off > dataSize) {
        assert(false);
        return;
    }
    size_t endOff = (size == (size_t)-1 ? dataSize : off + size);
    if (endOff > dataSize) {
        assert(false);
        endOff = dataSize;
    }
    size = endOff - off;
    if (dataSize == 0 || size == 0) {
        return;
    }

    volatile const char *c = (const char*)data + off, *e = c + size;
    for (; c < e; c += 512) {
        *c;
    }
}

class TMemoryMap::TImpl: public TAtomicRefCount<TImpl> {
public:
    inline void CreateMapping() {
#if defined(_win_)
        Mapping_ = nullptr;
        if (Length_) {
            Mapping_ = CreateFileMapping(File_.GetHandle(), nullptr,
                                         (Mode_ & oAccessMask) == TFileMap::oRdWr ? PAGE_READWRITE : PAGE_READONLY,
                                         (DWORD)(Length_ >> 32), (DWORD)(Length_ & 0xFFFFFFFF), nullptr);
            if (Mapping_ == nullptr) {
                ythrow yexception() << "Can't create file mapping of '" << DbgName_ << "': " << LastSystemErrorText();
            }
        } else {
            Mapping_ = MAP_FAILED;
        }
#elif defined(_unix_)
        if (!(Mode_ & oNotGreedy)) {
            PtrStart_ = mmap((caddr_t) nullptr, Length_, ModeToMmapProt(Mode_), ModeToMmapFlags(Mode_), File_.GetHandle(), 0);

            if ((MAP_FAILED == PtrStart_) && Length_) {
                ythrow yexception() << "Can't map " << (unsigned long)Length_ << " bytes of file '" << DbgName_ << "' at offset 0: " << LastSystemErrorText();
            }
        } else {
            PtrStart_ = nullptr;
        }
#endif
    }

    void CheckFile() const {
        if (!File_.IsOpen()) {
            ythrow yexception() << "TMemoryMap: FILE '" << DbgName_ << "' is not open, " << LastSystemErrorText();
        }
        if (Length_ < 0) {
            ythrow yexception() << "'" << DbgName_ << "' is not a regular file";
        }
    }

    inline TImpl(FILE* f, EOpenMode om, TString dbgName)
        : File_(Duplicate(f))
        , DbgName_(std::move(dbgName))
        , Length_(File_.GetLength())
        , Mode_(om)
    {
        CheckFile();
        CreateMapping();
    }

    inline TImpl(const TString& name, EOpenMode om)
        : File_(name, (om & oRdWr) ? OpenExisting | RdWr : OpenExisting | RdOnly)
        , DbgName_(name)
        , Length_(File_.GetLength())
        , Mode_(om)
    {
        CheckFile();
        CreateMapping();
    }

    inline TImpl(const TString& name, i64 length, EOpenMode om)
        : File_(name, (om & oRdWr) ? OpenExisting | RdWr : OpenExisting | RdOnly)
        , DbgName_(name)
        , Length_(length)
        , Mode_(om)
    {
        CheckFile();

        if (File_.GetLength() < Length_) {
            File_.Resize(Length_);
        }

        CreateMapping();
    }

    inline TImpl(const TFile& file, EOpenMode om, const TString& dbgName)
        : File_(file)
        , DbgName_(File_.GetName() ? File_.GetName() : dbgName)
        , Length_(File_.GetLength())
        , Mode_(om)
    {
        CheckFile();
        CreateMapping();
    }

    inline bool IsOpen() const noexcept {
        return File_.IsOpen()
#if defined(_win_)
               && Mapping_ != nullptr
#endif
            ;
    }

    inline bool IsWritable() const noexcept {
        return (Mode_ & oRdWr || Mode_ & oCopyOnWr);
    }

    inline TMapResult Map(i64 offset, size_t size) {
        assert(File_.IsOpen());

        if (offset > Length_) {
            ythrow yexception() << "Can't map something at offset " << offset << " of '" << DbgName_ << "' with length " << Length_;
        }

        if (offset + (i64)size > Length_) {
            ythrow yexception() << "Can't map " << (unsigned long)size << " bytes at offset " << offset << " of '" << DbgName_ << "' with length " << Length_;
        }

        TMapResult result;

        i64 base = DownToGranularity(offset);
        result.Head = (i32)(offset - base);
        size += result.Head;

#if defined(_win_)
        result.Ptr = MapViewOfFile(Mapping_,
                                   (Mode_ & oAccessMask) == oRdOnly ? FILE_MAP_READ : (Mode_ & oAccessMask) == oCopyOnWr ? FILE_MAP_COPY
                                                                                                                         : FILE_MAP_WRITE,
                                   Hi32(base), Lo32(base), size);
#else
    #if defined(_unix_)
        if (Mode_ & oNotGreedy) {
    #endif
            result.Ptr = mmap((caddr_t) nullptr, size, ModeToMmapProt(Mode_), ModeToMmapFlags(Mode_), File_.GetHandle(), base);

            if (result.Ptr == (char*)(-1)) {
                result.Ptr = nullptr;
            }
    #if defined(_unix_)
        } else {
            result.Ptr = PtrStart_ ? static_cast<caddr_t>(PtrStart_) + base : nullptr;
        }
    #endif
#endif
        if (result.Ptr != nullptr || size == 0) { // allow map of size 0
            result.Size = size;
        } else {
            ythrow yexception() << "Can't map " << (unsigned long)size << " bytes at offset " << offset << " of '" << DbgName_ << "': " << LastSystemErrorText();
        }
        NSan::Unpoison(result.Ptr, result.Size);
        if (Mode_ & oPrecharge) {
            NPrivate::Precharge(result.Ptr, result.Size, 0, result.Size);
        }

        return result;
    }

#if defined(_win_)
    inline bool Unmap(void* ptr, size_t) {
        return ::UnmapViewOfFile(ptr) != FALSE;
    }
#else
    inline bool Unmap(void* ptr, size_t size) {
    #if defined(_unix_)
        if (Mode_ & oNotGreedy) {
    #endif
            return size == 0 || ::munmap(static_cast<caddr_t>(ptr), size) == 0;
    #if defined(_unix_)
        } else {
            return true;
        }
    #endif
    }
#endif

    void SetSequential() {
#if defined(_unix_)
        if (!(Mode_ & oNotGreedy) && Length_) {
            MadviseSequentialAccess(PtrStart_, Length_);
        }
#endif
    }

    void Evict(void* ptr, size_t len) {
        MadviseEvict(ptr, len);
    }

    void Evict() {
#if defined(_unix_)
// Evict(PtrStart_, Length_);
#endif
    }

    inline ~TImpl() {
#if defined(_win_)
        if (Mapping_) {
            ::CloseHandle(Mapping_); // != FALSE
            Mapping_ = nullptr;
        }
#elif defined(_unix_)
        if (PtrStart_) {
            munmap((caddr_t)PtrStart_, Length_);
        }
#endif
    }

    inline i64 Length() const noexcept {
        return Length_;
    }

    inline TFile GetFile() const noexcept {
        return File_;
    }

    inline TString GetDbgName() const {
        return DbgName_;
    }

    inline EOpenMode GetMode() const noexcept {
        return Mode_;
    }

private:
    TFile File_;
    TString DbgName_; // This string is never used to actually open a file, only in exceptions
    i64 Length_;
    EOpenMode Mode_;

#if defined(_win_)
    void* Mapping_;
#elif defined(_unix_)
    void* PtrStart_;
#endif
};

TMemoryMap::TMemoryMap(const TString& name)
    : Impl_(new TImpl(name, EOpenModeFlag::oRdOnly))
{
}

TMemoryMap::TMemoryMap(const TString& name, EOpenMode om)
    : Impl_(new TImpl(name, om))
{
}

TMemoryMap::TMemoryMap(const TString& name, i64 length, EOpenMode om)
    : Impl_(new TImpl(name, length, om))
{
}

TMemoryMap::TMemoryMap(FILE* f, TString dbgName)
    : Impl_(new TImpl(f, EOpenModeFlag::oRdOnly, std::move(dbgName)))
{
}

TMemoryMap::TMemoryMap(FILE* f, EOpenMode om, TString dbgName)
    : Impl_(new TImpl(f, om, std::move(dbgName)))
{
}

TMemoryMap::TMemoryMap(const TFile& file, const TString& dbgName)
    : Impl_(new TImpl(file, EOpenModeFlag::oRdOnly, dbgName))
{
}

TMemoryMap::TMemoryMap(const TFile& file, EOpenMode om, const TString& dbgName)
    : Impl_(new TImpl(file, om, dbgName))
{
}

TMemoryMap::~TMemoryMap() = default;

TMemoryMap::TMapResult TMemoryMap::Map(i64 offset, size_t size) {
    return Impl_->Map(offset, size);
}

bool TMemoryMap::Unmap(void* ptr, size_t size) {
    return Impl_->Unmap(ptr, size);
}

bool TMemoryMap::Unmap(TMapResult region) {
    return Unmap(region.Ptr, region.Size);
}

void TMemoryMap::ResizeAndReset(i64 size) {
    EOpenMode om = Impl_->GetMode();
    TFile file = GetFile();
    file.Resize(size);
    Impl_.Reset(new TImpl(file, om, Impl_->GetDbgName()));
}

TMemoryMap::TMapResult TMemoryMap::ResizeAndRemap(i64 offset, size_t size) {
    ResizeAndReset(offset + (i64)size);
    return Map(offset, size);
}

void TMemoryMap::SetSequential() {
    Impl_->SetSequential();
}

void TMemoryMap::Evict(void* ptr, size_t len) {
    Impl_->Evict(ptr, len);
}

void TMemoryMap::Evict() {
    Impl_->Evict();
}

i64 TMemoryMap::Length() const noexcept {
    return Impl_->Length();
}

bool TMemoryMap::IsOpen() const noexcept {
    return Impl_->IsOpen();
}

bool TMemoryMap::IsWritable() const noexcept {
    return Impl_->IsWritable();
}

TMemoryMap::EOpenMode TMemoryMap::GetMode() const noexcept {
    return Impl_->GetMode();
}

TFile TMemoryMap::GetFile() const noexcept {
    return Impl_->GetFile();
}

TFileMap::TFileMap(const TMemoryMap& map) noexcept
    : Map_(map)
{
}

TFileMap::TFileMap(const TString& name)
    : Map_(name)
{
}

TFileMap::TFileMap(const TString& name, EOpenMode om)
    : Map_(name, om)
{
}

TFileMap::TFileMap(const TString& name, i64 length, EOpenMode om)
    : Map_(name, length, om)
{
}

TFileMap::TFileMap(FILE* f, EOpenMode om, TString dbgName)
    : Map_(f, om, std::move(dbgName))
{
}

TFileMap::TFileMap(const TFile& file, EOpenMode om, const TString& dbgName)
    : Map_(file, om, dbgName)
{
}

TFileMap::TFileMap(const TFileMap& fm) noexcept
    : Map_(fm.Map_)
{
}

void TFileMap::Flush(void* ptr, size_t size, bool sync) {
    Y_ASSERT(ptr >= Ptr());
    Y_ASSERT(static_cast<char*>(ptr) + size <= static_cast<char*>(Ptr()) + MappedSize());

    if (!Region_.IsMapped()) {
        return;
    }

#if defined(_win_)
    if (sync) {
        FlushViewOfFile(ptr, size);
    }
#else
    msync(ptr, size, sync ? MS_SYNC : MS_ASYNC);
#endif
}

TFileMap::TMapResult TFileMap::Map(i64 offset, size_t size) {
    Unmap();
    Region_ = Map_.Map(offset, size);
    return Region_;
}

TFileMap::TMapResult TFileMap::ResizeAndRemap(i64 offset, size_t size) {
    // explicit Unmap() is required because in oNotGreedy mode the Map_ object doesn't own the mapped area
    Unmap();
    Region_ = Map_.ResizeAndRemap(offset, size);
    return Region_;
}

void TFileMap::Unmap() {
    if (!Region_.IsMapped()) {
        return;
    }

    if (Map_.Unmap(Region_)) {
        Region_.Reset();
    } else {
        ythrow yexception() << "can't unmap file";
    }
}

TFileMap::~TFileMap() {
    try {
        // explicit Unmap() is required because in oNotGreedy mode the Map_ object doesn't own the mapped area
        Unmap();
    } catch (...) {
        // ¯\_(ツ)_/¯
    }
}

void TFileMap::Precharge(size_t pos, size_t size) const {
    NPrivate::Precharge(Ptr(), MappedSize(), pos, size);
}

TMappedAllocation::TMappedAllocation(size_t size, bool shared, void* addr)
    : Ptr_(nullptr)
    , Size_(0)
    , Shared_(shared)
#if defined(_win_)
    , Mapping_(nullptr)
#endif
{
    if (size != 0) {
        Alloc(size, addr);
    }
}

void* TMappedAllocation::Alloc(size_t size, void* addr) {
    assert(Ptr_ == nullptr);
#if defined(_win_)
    (void)addr;
    Mapping_ = CreateFileMapping((HANDLE)-1, nullptr, PAGE_READWRITE, 0, size ? size : 1, nullptr);
    Ptr_ = MapViewOfFile(Mapping_, FILE_MAP_WRITE, 0, 0, size ? size : 1);
#else
    Ptr_ = mmap(addr, size, PROT_READ | PROT_WRITE, (Shared_ ? MAP_SHARED : MAP_PRIVATE) | MAP_ANON, -1, 0);

    if (Ptr_ == (void*)MAP_FAILED) {
        Ptr_ = nullptr;
    }
#endif
    if (Ptr_ != nullptr) {
        Size_ = size;
    }
    return Ptr_;
}

void TMappedAllocation::Dealloc() {
    if (Ptr_ == nullptr) {
        return;
    }
#if defined(_win_)
    UnmapViewOfFile(Ptr_);
    CloseHandle(Mapping_);
    Mapping_ = nullptr;
#else
    munmap((caddr_t)Ptr_, Size_);
#endif
    Ptr_ = nullptr;
    Size_ = 0;
}

void TMappedAllocation::swap(TMappedAllocation& with) noexcept {
    DoSwap(Ptr_, with.Ptr_);
    DoSwap(Size_, with.Size_);
#if defined(_win_)
    DoSwap(Mapping_, with.Mapping_);
#endif
}
