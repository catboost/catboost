#pragma once

#include <util/generic/fwd.h>
#include <util/generic/utility.h>
#include <util/system/defaults.h>

class TMemoryMap;
class IInputStream;
class TFile;
class TBuffer;

/// @addtogroup BLOBs
/// @{
class TBlob {
public:
    class TBase {
    public:
        inline TBase() noexcept = default;
        virtual ~TBase() = default;

        virtual void Ref() noexcept = 0;
        virtual void UnRef() noexcept = 0;
    };

private:
    struct TStorage {
        const void* Data;
        size_t Length;
        TBase* Base;

        inline TStorage(const void* data, size_t length, TBase* base) noexcept
            : Data(data)
            , Length(length)
            , Base(base)
        {
        }

        inline ~TStorage() = default;

        inline void Swap(TStorage& r) noexcept {
            DoSwap(Data, r.Data);
            DoSwap(Length, r.Length);
            DoSwap(Base, r.Base);
        }
    };

public:
    using value_type = unsigned char;
    using const_reference = const value_type&;
    using const_pointer = const value_type*;
    using const_iterator = const_pointer;

    /**
     * Constructs a null blob (data array points to nullptr).
     */
    TBlob() noexcept
        : S_(nullptr, 0, nullptr)
    {
    }

    inline TBlob(const TBlob& r) noexcept
        : S_(r.S_)
    {
        Ref();
    }

    TBlob(TBlob&& r) noexcept
        : TBlob()
    {
        this->Swap(r);
    }

    inline TBlob(const void* data, size_t length, TBase* base) noexcept
        : S_(data, length, base)
    {
        Ref();
    }

    inline ~TBlob() {
        UnRef();
    }

    inline TBlob& operator=(const TBlob& r) noexcept {
        TBlob(r).Swap(*this);

        return *this;
    }

    /// Swaps content of two data arrays.
    inline void Swap(TBlob& r) noexcept {
        S_.Swap(r.S_);
    }

    /// Returns a const reference to the data array.
    inline const void* Data() const noexcept {
        return S_.Data;
    }

    /// Returns the size of the data array in bytes.
    inline size_t Length() const noexcept {
        return S_.Length;
    }

    /// Checks if the object has an empty data array.
    Y_PURE_FUNCTION
    inline bool Empty() const noexcept {
        return !Length();
    }

    /// Checks if the object has a data array.
    inline bool IsNull() const noexcept {
        return !Data();
    }

    /// Returns a const pointer of char type to the data array.
    inline const char* AsCharPtr() const noexcept {
        return (const char*)Data();
    }

    /// Returns a const pointer of unsigned char type to the data array.
    inline const unsigned char* AsUnsignedCharPtr() const noexcept {
        return (const unsigned char*)Data();
    }

    /// Drops the data array.
    inline void Drop() noexcept {
        TBlob().Swap(*this);
    }

    /*
     * Some stl-like methods
     */

    /// Returns the size of the data array in bytes.
    inline size_t Size() const noexcept {
        return Length();
    }

    /// Standard iterator.
    inline const_iterator Begin() const noexcept {
        return AsUnsignedCharPtr();
    }

    /// Standard iterator.
    inline const_iterator End() const noexcept {
        return Begin() + Size();
    }

    inline value_type operator[](size_t n) const noexcept {
        return *(Begin() + n);
    }

    /// Shortcut to SubBlob(0, len)
    TBlob SubBlob(size_t len) const;

    /// Creates a new object from the provided range [begin, end) of internal data. No memory allocation and no copy.
    /// @details Increments the refcounter of the current object
    TBlob SubBlob(size_t begin, size_t end) const;

    /// Calls Copy() for the internal data.
    TBlob DeepCopy() const;

    /// Creates a new blob with a single-threaded (non atomic) refcounter. Dynamically allocates memory and copies the data content.
    static TBlob CopySingleThreaded(const void* data, size_t length);

    /// Creates a new blob with a multi-threaded (atomic) refcounter. Dynamically allocates memory and copies the data content.
    static TBlob Copy(const void* data, size_t length);

    /// Creates a blob which doesn't own data. No refcounter, no memory allocation, no data copy.
    static TBlob NoCopy(const void* data, size_t length);

    /// Creates a blob with a single-threaded (non atomic) refcounter. It maps the file on the path as data.
    static TBlob FromFileSingleThreaded(const TString& path);

    /// Creates a blob with a multi-threaded (atomic) refcounter. It maps the file on the path as data.
    static TBlob FromFile(const TString& path);

    /// Creates a blob with a single-threaded (non atomic) refcounter. It maps the file on the path as data.
    static TBlob FromFileSingleThreaded(const TFile& file);

    /// Creates a blob with a multi-threaded (atomic) refcounter. It maps the file on the path as data.
    static TBlob FromFile(const TFile& file);

    // TODO: drop Precharged* functions.

    /// Creates a precharged blob with a single-threaded (non atomic) refcounter. It maps the file on the path as data.
    static TBlob PrechargedFromFileSingleThreaded(const TString& path);

    /// Creates a precharged blob with a multi-threaded (atomic) refcounter. It maps the file on the path as data.
    static TBlob PrechargedFromFile(const TString& path);

    /// Creates a precharged blob with a single-threaded (non atomic) refcounter. It maps the file content as data.
    static TBlob PrechargedFromFileSingleThreaded(const TFile& file);

    /// Creates a precharged blob with a multi-threaded (atomic) refcounter. It maps the file content as data.
    static TBlob PrechargedFromFile(const TFile& file);

    /// Creates a locked blob with a single-threaded (non atomic) refcounter. It maps the file on the path as data.
    static TBlob LockedFromFileSingleThreaded(const TString& path);

    /// Creates a locked blob with a multi-threaded (atomic) refcounter. It maps the file on the path as data.
    static TBlob LockedFromFile(const TString& path);

    /// Creates a locked blob with a single-threaded (non atomic) refcounter. It maps the file content as data.
    static TBlob LockedFromFileSingleThreaded(const TFile& file);

    /// Creates a locked blob with a multi-threaded (atomic) refcounter. It maps the file content as data.
    static TBlob LockedFromFile(const TFile& file);

    /// Creates a locked blob with a single-threaded (non atomic) refcounter from the mapped memory.
    static TBlob LockedFromMemoryMapSingleThreaded(const TMemoryMap& map, ui64 offset, size_t length);

    /// Creates a locked blob with a multi-threaded (atomic) refcounter from the mapped memory.
    static TBlob LockedFromMemoryMap(const TMemoryMap& map, ui64 offset, size_t length);

    /// Creates a blob with a single-threaded (non atomic) refcounter from the mapped memory.
    static TBlob FromMemoryMapSingleThreaded(const TMemoryMap& map, ui64 offset, size_t length);

    /// Creates a blob with a multi-threaded (atomic) refcounter from the mapped memory.
    static TBlob FromMemoryMap(const TMemoryMap& map, ui64 offset, size_t length);

    /// Creates a blob with a single-threaded (non atomic) refcounter. Dynamically allocates memory and copies data from the file on the path using pread().
    static TBlob FromFileContentSingleThreaded(const TString& path);

    /// Creates a blob with a multi-threaded (atomic) refcounter. Dynamically allocates memory and copies data from the file on the path using pread().
    static TBlob FromFileContent(const TString& path);

    /// Creates a blob with a single-threaded (non atomic) refcounter. Dynamically allocates memory and copies data from the file using pread().
    static TBlob FromFileContentSingleThreaded(const TFile& file);

    /// Creates a blob with a multi-threaded (atomic) refcounter. Dynamically allocates memory and copies data from the file using pread().
    static TBlob FromFileContent(const TFile& file);

    /// Creates a blob with a single-threaded (non atomic) refcounter. Dynamically allocates memory and copies data from the provided range of the file content using pread().
    static TBlob FromFileContentSingleThreaded(const TFile& file, ui64 offset, size_t length);

    /// Creates a blob with a multi-threaded (atomic) refcounter. Dynamically allocates memory and copies data from the provided range of the file content using pread().
    static TBlob FromFileContent(const TFile& file, ui64 offset, size_t length);

    /// Creates a blob from the stream content with a single-threaded (non atomic) refcounter.
    static TBlob FromStreamSingleThreaded(IInputStream& in);

    /// Creates a blob from the stream content with a multi-threaded (atomic) refcounter.
    static TBlob FromStream(IInputStream& in);

    /// Creates a blob with a single-threaded (non atomic) refcounter. No memory allocation, no content copy.
    /// @details The input object becomes empty.
    static TBlob FromBufferSingleThreaded(TBuffer& in);

    /// Creates a blob with a multi-threaded (atomic) refcounter. No memory allocation, no content copy.
    /// @details The input object becomes empty.
    static TBlob FromBuffer(TBuffer& in);

    /// Creates a blob from TString with a single-threaded (non atomic) refcounter.
    static TBlob FromStringSingleThreaded(const TString& s);

    /// Creates a blob from TString with a single-threaded (non atomic) refcounter. Doesn't copy its content.
    static TBlob FromStringSingleThreaded(TString&& s);

    /// Creates a blob from TString with a multi-threaded (atomic) refcounter.
    static TBlob FromString(const TString& s);

    /// Creates a blob from TString with a multi-threaded (atomic) refcounter. Doesn't copy its content.
    static TBlob FromString(TString&& s);

private:
    inline void Ref() noexcept {
        if (S_.Base) {
            S_.Base->Ref();
        }
    }

    inline void UnRef() noexcept {
        if (S_.Base) {
            S_.Base->UnRef();
        }
    }

private:
    TStorage S_;
};

/// @}
