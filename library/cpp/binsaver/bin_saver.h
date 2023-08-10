#pragma once

#include "buffered_io.h"
#include "class_factory.h"

#include <library/cpp/containers/2d_array/2d_array.h>

#include <util/generic/hash_set.h>
#include <util/generic/buffer.h>
#include <util/generic/list.h>
#include <util/generic/maybe.h>
#include <util/generic/bitmap.h>
#include <util/generic/variant.h>
#include <util/generic/ylimits.h>
#include <util/memory/blob.h>
#include <util/digest/murmur.h>
#include <util/system/compiler.h>

#include <array>
#include <bitset>
#include <list>
#include <string>

#ifdef _MSC_VER
#pragma warning(disable : 4127)
#endif

enum ESaverMode {
    SAVER_MODE_READ = 1,
    SAVER_MODE_WRITE = 2,
    SAVER_MODE_WRITE_COMPRESSED = 3,
};

namespace NBinSaverInternals {
    // This lets explicitly control the overload resolution priority
    // The higher P means higher priority in overload resolution order
    template <int P>
    struct TOverloadPriority : TOverloadPriority <P-1> {
    };

    template <>
    struct TOverloadPriority<0> {
    };
}

//////////////////////////////////////////////////////////////////////////
struct IBinSaver {
public:
    typedef unsigned char chunk_id;
    typedef ui32 TStoredSize; // changing this will break compatibility

private:
    // This overload is required to avoid infinite recursion when overriding serialization in derived classes:
    // struct B {
    //  virtual int operator &(IBinSaver& f) {
    //      return 0;
    //  }
    // };
    //
    // struct D : B {
    //  int operator &(IBinSaver& f) override {
    //      f.Add(0, static_cast<B*>(this));
    //      return 0;
    //  }
    // };
    template <class T, typename = decltype(std::declval<T*>()->T::operator&(std::declval<IBinSaver&>()))>
    void CallObjectSerialize(T* p, NBinSaverInternals::TOverloadPriority<2>) { // highest priority -  will be resolved first if enabled
                                           // Note: p->operator &(*this) would lead to infinite recursion
        p->T::operator&(*this);
    }

    template <class T, typename = decltype(std::declval<T&>() & std::declval<IBinSaver&>())>
    void CallObjectSerialize(T* p, NBinSaverInternals::TOverloadPriority<1>) { // lower priority - will be resolved second if enabled
        (*p) & (*this);
    }

    template <class T>
    void CallObjectSerialize(T* p, NBinSaverInternals::TOverloadPriority<0>) { // lower priority - will be resolved last
#if (!defined(_MSC_VER))
        // broken in clang16 for some types
        // In MSVC __has_trivial_copy returns false to enums, primitive types and arrays.
        // static_assert(__is_trivially_copyable(T), "Class is nontrivial copyable, you must define operator&, see");
#endif
        DataChunk(p, sizeof(T));
    }

    // vector
    template <class T, class TA>
    void DoVector(TVector<T, TA>& data) {
        TStoredSize nSize;
        if (IsReading()) {
            data.clear();
            Add(2, &nSize);
            data.resize(nSize);
        } else {
            nSize = data.size();
            CheckOverflow(nSize, data.size());
            Add(2, &nSize);
        }
        for (TStoredSize i = 0; i < nSize; i++)
            Add(1, &data[i]);
    }

    template <class T, int N>
    void DoArray(T (&data)[N]) {
        for (size_t i = 0; i < N; i++) {
            Add(1, &(data[i]));
        }
    }

    template <typename TLarge>
    void CheckOverflow(TStoredSize nSize, TLarge origSize) {
        if (nSize != origSize) {
            fprintf(stderr, "IBinSaver: object size is too large to be serialized (%" PRIu32 " != %" PRIu64 ")\n", nSize, (ui64)origSize);
            abort();
        }
    }

    template <class T, class TA>
    void DoDataVector(TVector<T, TA>& data) {
        TStoredSize nSize = data.size();
        CheckOverflow(nSize, data.size());
        Add(1, &nSize);
        if (IsReading()) {
            data.clear();
            data.resize(nSize);
        }
        if (nSize > 0)
            DataChunk(&data[0], sizeof(T) * nSize);
    }

    template <class AM>
    void DoAnyMap(AM& data) {
        if (IsReading()) {
            data.clear();
            TStoredSize nSize;
            Add(3, &nSize);
            TVector<typename AM::key_type, typename std::allocator_traits<typename AM::allocator_type>::template rebind_alloc<typename AM::key_type>> indices;
            indices.resize(nSize);
            for (TStoredSize i = 0; i < nSize; ++i)
                Add(1, &indices[i]);
            for (TStoredSize i = 0; i < nSize; ++i)
                Add(2, &data[indices[i]]);
        } else {
            TStoredSize nSize = data.size();
            CheckOverflow(nSize, data.size());
            Add(3, &nSize);

            TVector<typename AM::key_type, typename std::allocator_traits<typename AM::allocator_type>::template rebind_alloc<typename AM::key_type>> indices;
            indices.resize(nSize);
            TStoredSize i = 1;
            for (auto pos = data.begin(); pos != data.end(); ++pos, ++i)
                indices[nSize - i] = pos->first;
            for (TStoredSize j = 0; j < nSize; ++j)
                Add(1, &indices[j]);
            for (TStoredSize j = 0; j < nSize; ++j)
                Add(2, &data[indices[j]]);
        }
    }

    // hash_multimap
    template <class AMM>
    void DoAnyMultiMap(AMM& data) {
        if (IsReading()) {
            data.clear();
            TStoredSize nSize;
            Add(3, &nSize);
            TVector<typename AMM::key_type, typename std::allocator_traits<typename AMM::allocator_type>::template rebind_alloc<typename AMM::key_type>> indices;
            indices.resize(nSize);
            for (TStoredSize i = 0; i < nSize; ++i)
                Add(1, &indices[i]);
            for (TStoredSize i = 0; i < nSize; ++i) {
                std::pair<typename AMM::key_type, typename AMM::mapped_type> valToInsert;
                valToInsert.first = indices[i];
                Add(2, &valToInsert.second);
                data.insert(valToInsert);
            }
        } else {
            TStoredSize nSize = data.size();
            CheckOverflow(nSize, data.size());
            Add(3, &nSize);
            for (auto pos = data.begin(); pos != data.end(); ++pos)
                Add(1, (typename AMM::key_type*)(&pos->first));
            for (auto pos = data.begin(); pos != data.end(); ++pos)
                Add(2, &pos->second);
        }
    }

    template <class T>
    void DoAnySet(T& data) {
        if (IsReading()) {
            data.clear();
            TStoredSize nSize;
            Add(2, &nSize);
            for (TStoredSize i = 0; i < nSize; ++i) {
                typename T::value_type member;
                Add(1, &member);
                data.insert(member);
            }
        } else {
            TStoredSize nSize = data.size();
            CheckOverflow(nSize, data.size());
            Add(2, &nSize);
            for (const auto& elem : data) {
                auto member = elem;
                Add(1, &member);
            }
        }
    }

    // 2D array
    template <class T>
    void Do2DArray(TArray2D<T>& a) {
        int nXSize = a.GetXSize(), nYSize = a.GetYSize();
        Add(1, &nXSize);
        Add(2, &nYSize);
        if (IsReading())
            a.SetSizes(nXSize, nYSize);
        for (int i = 0; i < nXSize * nYSize; i++)
            Add(3, &a[i / nXSize][i % nXSize]);
    }
    template <class T>
    void Do2DArrayData(TArray2D<T>& a) {
        int nXSize = a.GetXSize(), nYSize = a.GetYSize();
        Add(1, &nXSize);
        Add(2, &nYSize);
        if (IsReading())
            a.SetSizes(nXSize, nYSize);
        if (nXSize * nYSize > 0)
            DataChunk(&a[0][0], sizeof(T) * nXSize * nYSize);
    }
    // strings
    template <class TStringType>
    void DataChunkStr(TStringType& data, i64 elemSize) {
        if (bRead) {
            TStoredSize nCount = 0;
            File.Read(&nCount, sizeof(TStoredSize));
            data.resize(nCount);
            if (nCount)
                File.Read(&*data.begin(), nCount * elemSize);
        } else {
            TStoredSize nCount = data.size();
            CheckOverflow(nCount, data.size());
            File.Write(&nCount, sizeof(TStoredSize));
            File.Write(data.c_str(), nCount * elemSize);
        }
    }
    void DataChunkString(std::string& data) {
        DataChunkStr(data, sizeof(char));
    }
    void DataChunkStroka(TString& data) {
        DataChunkStr(data, sizeof(TString::char_type));
    }
    void DataChunkWtroka(TUtf16String& data) {
        DataChunkStr(data, sizeof(wchar16));
    }

    void DataChunk(void* pData, i64 nSize) {
        i64 chunkSize = 1 << 30;
        for (i64 offset = 0; offset < nSize; offset += chunkSize) {
            void* ptr = (char*)pData + offset;
            i64 size = offset + chunkSize < nSize ? chunkSize : (nSize - offset);
            if (bRead)
                File.Read(ptr, size);
            else
                File.Write(ptr, size);
        }
    }

    // storing/loading pointers to objects
    void StoreObject(IObjectBase* pObject);
    IObjectBase* LoadObject();

    bool bRead;
    TBufferedStream<> File;
    // maps objects addresses during save(first) to addresses during load(second) - during loading
    // or serves as a sign that some object has been already stored - during storing
    bool StableOutput;

    typedef THashMap<void*, ui32> PtrIdHash;
    TAutoPtr<PtrIdHash> PtrIds;

    typedef THashMap<ui64, TPtr<IObjectBase>> CObjectsHash;
    TAutoPtr<CObjectsHash> Objects;

    TVector<IObjectBase*> ObjectQueue;

public:
    bool IsReading() {
        return bRead;
    }
    void AddRawData(const chunk_id, void* pData, i64 nSize) {
        DataChunk(pData, nSize);
    }

    // return type of Add() is used to detect specialized serializer (see HasNonTrivialSerializer below)
    template <class T>
    char Add(const chunk_id, T* p) {
        CallObjectSerialize(p, NBinSaverInternals::TOverloadPriority<2>());
        return 0;
    }
    int Add(const chunk_id, std::string* pStr) {
        DataChunkString(*pStr);
        return 0;
    }
    int Add(const chunk_id, TString* pStr) {
        DataChunkStroka(*pStr);
        return 0;
    }
    int Add(const chunk_id, TUtf16String* pStr) {
        DataChunkWtroka(*pStr);
        return 0;
    }
    int Add(const chunk_id, TBlob* blob) {
        if (bRead) {
            ui64 size = 0;
            File.Read(&size, sizeof(size));
            TBuffer buffer;
            buffer.Advance(size);
            if (size > 0)
                File.Read(buffer.Data(), buffer.Size());
            (*blob) = TBlob::FromBuffer(buffer);
        } else {
            const ui64 size = blob->Size();
            File.Write(&size, sizeof(size));
            File.Write(blob->Data(), blob->Size());
        }
        return 0;
    }
    template <class T1, class TA>
    int Add(const chunk_id, TVector<T1, TA>* pVec) {
        if (HasNonTrivialSerializer<T1>(0u))
            DoVector(*pVec);
        else
            DoDataVector(*pVec);
        return 0;
    }

    template <class T, int N>
    int Add(const chunk_id, T (*pVec)[N]) {
        if (HasNonTrivialSerializer<T>(0u))
            DoArray(*pVec);
        else
            DataChunk(pVec, sizeof(*pVec));
        return 0;
    }

    template <class T1, class T2, class T3, class T4>
    int Add(const chunk_id, TMap<T1, T2, T3, T4>* pMap) {
        DoAnyMap(*pMap);
        return 0;
    }
    template <class T1, class T2, class T3, class T4, class T5>
    int Add(const chunk_id, THashMap<T1, T2, T3, T4, T5>* pHash) {
        DoAnyMap(*pHash);
        return 0;
    }
    template <class T1, class T2, class T3, class T4, class T5>
    int Add(const chunk_id, THashMultiMap<T1, T2, T3, T4, T5>* pHash) {
        DoAnyMultiMap(*pHash);
        return 0;
    }
    template <class K, class L, class A>
    int Add(const chunk_id, TSet<K, L, A>* pSet) {
        DoAnySet(*pSet);
        return 0;
    }
    template <class T1, class T2, class T3, class T4>
    int Add(const chunk_id, THashSet<T1, T2, T3, T4>* pHash) {
        DoAnySet(*pHash);
        return 0;
    }

    template <class T1>
    int Add(const chunk_id, TArray2D<T1>* pArr) {
        if (HasNonTrivialSerializer<T1>(0u))
            Do2DArray(*pArr);
        else
            Do2DArrayData(*pArr);
        return 0;
    }
    template <class T1>
    int Add(const chunk_id, TList<T1>* pList) {
        TList<T1>& data = *pList;
        if (IsReading()) {
            int nSize;
            Add(2, &nSize);
            data.clear();
            data.insert(data.begin(), nSize, T1());
        } else {
            int nSize = data.size();
            Add(2, &nSize);
        }
        int i = 1;
        for (typename TList<T1>::iterator k = data.begin(); k != data.end(); ++k, ++i)
            Add(i + 2, &(*k));
        return 0;
    }
    template <class T1, class T2>
    int Add(const chunk_id, std::pair<T1, T2>* pData) {
        Add(1, &(pData->first));
        Add(2, &(pData->second));
        return 0;
    }

    template <class T1, size_t N>
    int Add(const chunk_id, std::array<T1, N>* pData) {
        if (HasNonTrivialSerializer<T1>(0u)) {
            for (size_t i = 0; i < N; ++i)
                Add(1, &(*pData)[i]);
        } else {
            DataChunk((void*)pData->data(), pData->size() * sizeof(T1));
        }
        return 0;
    }

    template <size_t N>
    int Add(const chunk_id, std::bitset<N>* pData) {
        if (IsReading()) {
            std::string s;
            Add(1, &s);
            *pData = std::bitset<N>(s);
        } else {
            std::string s = pData->template to_string<char, std::char_traits<char>, std::allocator<char>>();
            Add(1, &s);
        }
        return 0;
    }

    int Add(const chunk_id, TDynBitMap* pData) {
        if (IsReading()) {
            ui64 count = 0;
            Add(1, &count);
            pData->Clear();
            pData->Reserve(count * sizeof(TDynBitMap::TChunk) * 8);
            for (ui64 i = 0; i < count; ++i) {
                TDynBitMap::TChunk chunk = 0;
                Add(i + 1, &chunk);
                if (i > 0) {
                    pData->LShift(8 * sizeof(TDynBitMap::TChunk));
                }
                pData->Or(chunk);
            }
        } else {
            ui64 count = pData->GetChunkCount();
            Add(1, &count);
            for (ui64 i = 0; i < count; ++i) {
                // Write in reverse order
                TDynBitMap::TChunk chunk = pData->GetChunks()[count - i - 1];
                Add(i + 1, &chunk);
            }
        }
        return 0;
    }

    template <class TVariantClass>
    struct TLoadFromTypeFromListHelper {
        template <class T0, class... TTail>
        static void Do(IBinSaver& binSaver, ui32 typeIndex, TVariantClass* pData) {
            if constexpr (sizeof...(TTail) == 0) {
                Y_ASSERT(typeIndex == 0);
                T0 chunk;
                binSaver.Add(2, &chunk);
                *pData = std::move(chunk);
            } else {
                if (typeIndex == 0) {
                    Do<T0>(binSaver, 0, pData);
                } else {
                    Do<TTail...>(binSaver, typeIndex - 1, pData);
                }
            }
        }
    };

    template <class... TVariantTypes>
    int Add(const chunk_id, std::variant<TVariantTypes...>* pData) {
        static_assert(std::variant_size_v<std::variant<TVariantTypes...>> < Max<ui32>());

        ui32 index;
        if (IsReading()) {
            Add(1, &index);
            TLoadFromTypeFromListHelper<std::variant<TVariantTypes...>>::template Do<TVariantTypes...>(
                *this,
                index,
                pData
            );
        } else {
            index = pData->index(); // type cast is safe because of static_assert check above
            Add(1, &index);
            std::visit([&](auto& dst) -> void { Add(2, &dst); }, *pData);
        }
        return 0;
    }


    void AddPolymorphicBase(chunk_id, IObjectBase* pObject) {
        (*pObject) & (*this);
    }

    template <class T1, class T2>
    void DoPtr(TPtrBase<T1, T2>* pData) {
        if (pData && pData->Get()) {
        }
        if (IsReading())
            pData->Set(CastToUserObject(LoadObject(), (T1*)nullptr));
        else
            StoreObject(pData->GetBarePtr());
    }
    template <class T, class TPolicy>
    int Add(const chunk_id, TMaybe<T, TPolicy>* pData) {
        TMaybe<T, TPolicy>& data = *pData;
        if (IsReading()) {
            bool defined = false;
            Add(1, &defined);
            if (defined) {
                data = T();
                Add(2, data.Get());
            }
        } else {
            bool defined = data.Defined();
            Add(1, &defined);
            if (defined) {
                Add(2, data.Get());
            }
        }
        return 0;
    }

    template <typename TOne>
    void AddMulti(TOne& one) {
        Add(0, &one);
    }

    template <typename THead, typename... TTail>
    void AddMulti(THead& head, TTail&... tail) {
        Add(0, &head);
        AddMulti(tail...);
    }

    template <class T, typename = decltype(std::declval<T&>() & std::declval<IBinSaver&>())>
    static bool HasNonTrivialSerializer(ui32) {
        return true;
    }

    template <class T>
    static bool HasNonTrivialSerializer(...) {
        return sizeof(std::declval<IBinSaver*>()->Add(0, std::declval<T*>())) != 1;
    }

public:
    IBinSaver(IBinaryStream& stream, bool _bRead, bool stableOutput = false)
        : bRead(_bRead)
        , File(_bRead, stream)
        , StableOutput(stableOutput)
    {
    }
    virtual ~IBinSaver();
    bool IsValid() const {
        return File.IsValid();
    }
};

// realisation of forward declared serialisation operator
template <class TUserObj, class TRef>
int TPtrBase<TUserObj, TRef>::operator&(IBinSaver& f) {
    f.DoPtr(this);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

extern TClassFactory<IObjectBase>* pSaverClasses;
void StartRegisterSaveload();

template <class TReg>
struct TRegisterSaveLoadType {
    TRegisterSaveLoadType(int num) {
        StartRegisterSaveload();
        pSaverClasses->RegisterType(num, TReg::NewSaveLoadNullItem, (TReg*)nullptr);
    }
};

#define Y_BINSAVER_REGISTER(name) \
    BASIC_REGISTER_CLASS(name)    \
    static TRegisterSaveLoadType<name> init##name(MurmurHash<int>(#name, sizeof(#name)));

#define REGISTER_SAVELOAD_CLASS(N, name) \
    BASIC_REGISTER_CLASS(name)           \
    static TRegisterSaveLoadType<name> init##name##N(N);

// using TObj/TRef on forward declared templ class will not work
// but multiple registration with same id is allowed
#define REGISTER_SAVELOAD_TEMPL1_CLASS(N, className, T) \
    static TRegisterSaveLoadType<className<T>> init##className##T##N(N);

#define REGISTER_SAVELOAD_TEMPL2_CLASS(N, className, T1, T2)    \
    typedef className<T1, T2> temp##className##T1##_##T2##temp; \
    static TRegisterSaveLoadType<className<T1, T2>> init##className##T1##_##T2##N(N);

#define REGISTER_SAVELOAD_TEMPL3_CLASS(N, className, T1, T2, T3)           \
    typedef className<T1, T2, T3> temp##className##T1##_##T2##_##T3##temp; \
    static TRegisterSaveLoadType<className<T1, T2, T3>> init##className##T1##_##T2##_##T3##N(N);

#define REGISTER_SAVELOAD_NM_CLASS(N, nmspace, className) \
    BASIC_REGISTER_CLASS(nmspace::className)              \
    static TRegisterSaveLoadType<nmspace::className> init_##nmspace##_##name##N(N);

#define REGISTER_SAVELOAD_NM2_CLASS(N, nmspace1, nmspace2, className) \
    BASIC_REGISTER_CLASS(nmspace1::nmspace2::className)              \
    static TRegisterSaveLoadType<nmspace1::nmspace2::className> init_##nmspace1##_##nmspace2##_##name##N(N);

#define REGISTER_SAVELOAD_TEMPL1_NM_CLASS(N, nmspace, className, T)       \
    typedef nmspace::className<T> temp_init##nmspace##className##T##temp; \
    BASIC_REGISTER_CLASS(nmspace::className<T>)                           \
    static TRegisterSaveLoadType<nmspace::className<T>> temp_init##nmspace##_##name##T##N(N);

#define REGISTER_SAVELOAD_CLASS_NAME(N, cls, name) \
    BASIC_REGISTER_CLASS(cls)                      \
    static TRegisterSaveLoadType<cls> init##name##N(N);

#define REGISTER_SAVELOAD_CLASS_NS_PREF(N, cls, ns, pref) \
    REGISTER_SAVELOAD_CLASS_NAME(N, ns ::cls, _##pref##_##cls)

#define SAVELOAD(...)             \
    int operator&(IBinSaver& f) { \
        f.AddMulti(__VA_ARGS__);  \
        return 0;                 \
    } Y_SEMICOLON_GUARD

#define SAVELOAD_OVERRIDE_WITHOUT_BASE(...) \
    int operator&(IBinSaver& f) override {  \
        f.AddMulti(__VA_ARGS__);            \
        return 0;                           \
    } Y_SEMICOLON_GUARD

#define SAVELOAD_OVERRIDE(base, ...)       \
    int operator&(IBinSaver& f) override { \
        base::operator&(f);                \
        f.AddMulti(__VA_ARGS__);           \
        return 0;                          \
    } Y_SEMICOLON_GUARD

#define SAVELOAD_BASE(...)        \
    int operator&(IBinSaver& f) { \
        TBase::operator&(f);      \
        f.AddMulti(__VA_ARGS__);  \
        return 0;                 \
    } Y_SEMICOLON_GUARD
