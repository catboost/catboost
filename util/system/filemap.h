#pragma once

#include "file.h"
#include "align.h"
#include "yassert.h"

#include <util/generic/noncopyable.h>
#include <util/generic/ptr.h>
#include <util/generic/utility.h>
#include <util/generic/yexception.h>
#include <util/generic/flags.h>
#include <util/generic/string.h>

#include <new>
#include <cstdio>

namespace NPrivate {
    // NB: use TFileMap::Precharge() and TFileMappedArray::Prechage()
    void Precharge(const void* data, size_t dataSize, size_t offset, size_t size);
}

struct TMemoryMapCommon {
    struct TMapResult {
        inline size_t MappedSize() const noexcept {
            return Size - Head;
        }

        inline void* MappedData() const noexcept {
            return Ptr ? (void*)((char*)Ptr + Head) : nullptr;
        }

        inline bool IsMapped() const noexcept {
            return Ptr != nullptr;
        }

        inline void Reset() noexcept {
            Ptr = nullptr;
            Size = 0;
            Head = 0;
        }

        void* Ptr;
        size_t Size;
        i32 Head;

        TMapResult(void) noexcept {
            Reset();
        }
    };

    enum EOpenModeFlag {
        oRdOnly = 1,
        oRdWr = 2,
        oCopyOnWr = 4,

        oAccessMask = 7,
        oNotGreedy = 8,
        oPrecharge = 16,
        oPopulate = 32, // Populate page table entries (see mmap's MAP_POPULATE)
    };
    Y_DECLARE_FLAGS(EOpenMode, EOpenModeFlag);

    /**
     * Name that will be printed in exceptions if not specified.
     * Overridden by name obtained from `TFile` if it's not empty.
     */
    static const TString& UnknownFileName();
};
Y_DECLARE_OPERATORS_FOR_FLAGS(TMemoryMapCommon::EOpenMode);

class TMemoryMap: public TMemoryMapCommon {
public:
    explicit TMemoryMap(const TString& name);
    explicit TMemoryMap(const TString& name, EOpenMode om);
    TMemoryMap(const TString& name, i64 length, EOpenMode om);
    TMemoryMap(FILE* f, TString dbgName = UnknownFileName());
    TMemoryMap(FILE* f, EOpenMode om, TString dbgName = UnknownFileName());
    TMemoryMap(const TFile& file, const TString& dbgName = UnknownFileName());
    TMemoryMap(const TFile& file, EOpenMode om, const TString& dbgName = UnknownFileName());

    ~TMemoryMap();

    TMapResult Map(i64 offset, size_t size);
    bool Unmap(TMapResult region);

    void ResizeAndReset(i64 size);
    TMapResult ResizeAndRemap(i64 offset, size_t size);

    i64 Length() const noexcept;
    bool IsOpen() const noexcept;
    bool IsWritable() const noexcept;
    EOpenMode GetMode() const noexcept;
    TFile GetFile() const noexcept;

    void SetSequential();
    void Evict(void* ptr, size_t len);
    void Evict();

    /*
     * deprecated
     */
    bool Unmap(void* ptr, size_t size);

private:
    class TImpl;
    TSimpleIntrusivePtr<TImpl> Impl_;
};

class TFileMap: public TMemoryMapCommon {
public:
    TFileMap(const TMemoryMap& map) noexcept;
    TFileMap(const TString& name);
    TFileMap(const TString& name, EOpenMode om);
    TFileMap(const TString& name, i64 length, EOpenMode om);
    TFileMap(FILE* f, EOpenMode om = oRdOnly, TString dbgName = UnknownFileName());
    TFileMap(const TFile& file, EOpenMode om = oRdOnly, const TString& dbgName = UnknownFileName());
    TFileMap(const TFileMap& fm) noexcept;

    ~TFileMap();

    TMapResult Map(i64 offset, size_t size);
    TMapResult ResizeAndRemap(i64 offset, size_t size);
    void Unmap();

    void Flush(void* ptr, size_t size) {
        Flush(ptr, size, true);
    }

    void Flush() {
        Flush(Ptr(), MappedSize());
    }

    void FlushAsync(void* ptr, size_t size) {
        Flush(ptr, size, false);
    }

    void FlushAsync() {
        FlushAsync(Ptr(), MappedSize());
    }

    inline i64 Length() const noexcept {
        return Map_.Length();
    }

    inline bool IsOpen() const noexcept {
        return Map_.IsOpen();
    }

    inline bool IsWritable() const noexcept {
        return Map_.IsWritable();
    }

    EOpenMode GetMode() const noexcept {
        return Map_.GetMode();
    }

    inline void* Ptr() const noexcept {
        return Region_.MappedData();
    }

    inline size_t MappedSize() const noexcept {
        return Region_.MappedSize();
    }

    TFile GetFile() const noexcept {
        return Map_.GetFile();
    }

    void Precharge(size_t pos = 0, size_t size = (size_t)-1) const;

    void SetSequential() {
        Map_.SetSequential();
    }

    void Evict() {
        Map_.Evict();
    }

private:
    void Flush(void* ptr, size_t size, bool sync);

    TMemoryMap Map_;
    TMapResult Region_;
};

template <class T>
class TFileMappedArray {
private:
    const T* Ptr_;
    const T* End_;
    size_t Size_;
    char DummyData_[sizeof(T) + PLATFORM_DATA_ALIGN];
    mutable THolder<T, TDestructor> Dummy_;
    THolder<TFileMap> DataHolder_;

public:
    TFileMappedArray()
        : Ptr_(nullptr)
        , End_(nullptr)
        , Size_(0)
    {
    }
    ~TFileMappedArray() {
        Ptr_ = nullptr;
        End_ = nullptr;
    }
    void Init(const char* name) {
        DataHolder_.Reset(new TFileMap(name));
        DoInit(name);
    }
    void Init(const TFileMap& fileMap) {
        DataHolder_.Reset(new TFileMap(fileMap));
        DoInit(fileMap.GetFile().GetName());
    }
    void Term() {
        DataHolder_.Destroy();
        Ptr_ = nullptr;
        Size_ = 0;
        End_ = nullptr;
    }
    void Precharge() {
        DataHolder_->Precharge();
    }
    const T& operator[](size_t pos) const {
        Y_ASSERT(pos < size());
        return Ptr_[pos];
    }
    /// for STL compatibility only, Size() usage is recommended
    size_t size() const {
        return Size_;
    }
    size_t Size() const {
        return Size_;
    }
    const T& GetAt(size_t pos) const {
        if (pos < Size_)
            return Ptr_[pos];
        return Dummy();
    }
    void SetDummy(const T& n_Dummy) {
        Dummy_.Destroy();
        Dummy_.Reset(new (DummyData()) T(n_Dummy));
    }
    inline char* DummyData() const noexcept {
        return AlignUp((char*)DummyData_);
    }
    inline const T& Dummy() const {
        if (!Dummy_) {
            Dummy_.Reset(new (DummyData()) T());
        }

        return *Dummy_;
    }
    /// for STL compatibility only, Empty() usage is recommended
    Y_PURE_FUNCTION bool empty() const noexcept {
        return Empty();
    }

    Y_PURE_FUNCTION bool Empty() const noexcept {
        return 0 == Size_;
    }
    /// for STL compatibility only, Begin() usage is recommended
    const T* begin() const noexcept {
        return Begin();
    }
    const T* Begin() const noexcept {
        return Ptr_;
    }
    /// for STL compatibility only, End() usage is recommended
    const T* end() const noexcept {
        return End_;
    }
    const T* End() const noexcept {
        return End_;
    }

private:
    void DoInit(const TString& fileName) {
        DataHolder_->Map(0, DataHolder_->Length());
        if (DataHolder_->Length() % sizeof(T)) {
            Term();
            ythrow yexception() << "Incorrect size of file " << fileName.Quote();
        }
        Ptr_ = (const T*)DataHolder_->Ptr();
        Size_ = DataHolder_->Length() / sizeof(T);
        End_ = Ptr_ + Size_;
    }
};

class TMappedAllocation: TMoveOnly {
public:
    TMappedAllocation(size_t size = 0, bool shared = false, void* addr = nullptr);
    ~TMappedAllocation() {
        Dealloc();
    }
    TMappedAllocation(TMappedAllocation&& other) noexcept {
        this->swap(other);
    }
    TMappedAllocation& operator=(TMappedAllocation&& other) noexcept {
        this->swap(other);
        return *this;
    }
    void* Alloc(size_t size, void* addr = nullptr);
    void Dealloc();
    void* Ptr() const {
        return Ptr_;
    }
    char* Data(ui32 pos = 0) const {
        return (char*)(Ptr_ ? ((char*)Ptr_ + pos) : nullptr);
    }
    char* Begin() const noexcept {
        return (char*)Ptr();
    }
    char* End() const noexcept {
        return Begin() + MappedSize();
    }
    size_t MappedSize() const {
        return Size_;
    }
    void swap(TMappedAllocation& with) noexcept;

private:
    void* Ptr_ = nullptr;
    size_t Size_ = 0;
    bool Shared_ = false;
#ifdef _win_
    void* Mapping_ = nullptr;
#endif
};

template <class T>
class TMappedArray: private TMappedAllocation {
public:
    TMappedArray(size_t siz = 0)
        : TMappedAllocation(0)
    {
        if (siz)
            Create(siz);
    }
    ~TMappedArray() {
        Destroy();
    }
    T* Create(size_t siz) {
        Y_ASSERT(MappedSize() == 0 && Ptr() == nullptr);
        T* arr = (T*)Alloc((sizeof(T) * siz));
        if (!arr)
            return nullptr;
        Y_ASSERT(MappedSize() == sizeof(T) * siz);
        for (size_t n = 0; n < siz; n++)
            new (&arr[n]) T();
        return arr;
    }
    void Destroy() {
        T* arr = (T*)Ptr();
        if (arr) {
            for (size_t n = 0; n < size(); n++)
                arr[n].~T();
            Dealloc();
        }
    }
    T& operator[](size_t pos) {
        Y_ASSERT(pos < size());
        return ((T*)Ptr())[pos];
    }
    const T& operator[](size_t pos) const {
        Y_ASSERT(pos < size());
        return ((T*)Ptr())[pos];
    }
    T* begin() {
        return (T*)Ptr();
    }
    T* end() {
        return (T*)((char*)Ptr() + MappedSize());
    }
    size_t size() const {
        return MappedSize() / sizeof(T);
    }
    void swap(TMappedArray<T>& with) {
        TMappedAllocation::swap(with);
    }
};
