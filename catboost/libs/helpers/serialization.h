#pragma once

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>
#include <util/stream/length.h>
#include <util/system/yassert.h>


namespace NJson {
    class TJsonValue;
}


namespace NCB {

    // just to avoid repeating writing const_cast
    template <typename TOne>
    void SaveMulti(IBinSaver* binSaver, const TOne& one) {
        Y_ASSERT(!binSaver->IsReading());
        binSaver->Add(0, const_cast<TOne*>(&one));
    }

    template <typename THead, typename... TTail>
    void SaveMulti(IBinSaver* binSaver, const THead& head, const TTail&... tail) {
        Y_ASSERT(!binSaver->IsReading());
        binSaver->Add(0, const_cast<THead*>(&head));
        SaveMulti(binSaver, tail...);
    }

    // for symmetry with SaveMulti
    template <typename TOne>
    void LoadMulti(IBinSaver* binSaver, TOne* one) {
        Y_ASSERT(binSaver->IsReading());
        binSaver->Add(0, one);
    }

    template <typename THead, typename... TTail>
    void LoadMulti(IBinSaver* binSaver, THead* head, TTail*... tail) {
        Y_ASSERT(binSaver->IsReading());
        binSaver->Add(0, head);
        LoadMulti(binSaver, tail...);
    }

    // Note: does not store data size!
    template <class T>
    inline void SaveArrayData(TConstArrayRef<T> data, IBinSaver* binSaver) {
        Y_ASSERT(!binSaver->IsReading());
        if (IBinSaver::HasNonTrivialSerializer<T>(0u)) {
            for (const auto& element : data) {
                binSaver->Add(0, const_cast<T*>(&element));
            }
        } else {
            binSaver->AddRawData(
                0,
                const_cast<void*>((const void*)data.data()),
                SafeIntegerCast<i64>(sizeof(T) * data.size())
            );
        }
    }

    // Note: does not load data size!
    template <class T>
    inline void LoadArrayData(TArrayRef<T> data, IBinSaver* binSaver) {
        Y_ASSERT(binSaver->IsReading());
        if (IBinSaver::HasNonTrivialSerializer<T>(0u)) {
            for (auto& element : data) {
                binSaver->Add(0, &element);
            }
        } else {
            binSaver->AddRawData(0, (void*)data.data(), SafeIntegerCast<i64>(sizeof(T) * data.size()));
        }
    }


    /* Other smart pointer is incompatible with IObjectsBase/TObj, so we need separate methods.
     *  Note that unlike with IObjectsBase/TObj this serializer does not need OBJECT_NOCOPY_METHODS and
     *  REGISTER_SAVELOAD*, but it requires default constructor and operator& for T as usual.
     *
     *  Note that each instance of these pointers will be saved and loaded separately,
     *   no keeping track of sharing while serializing.
     *
     *  For use inside classes' operator& implementations (see serialization_ut.cpp for example)
     */
    namespace NPrivate {

        template <class TPtr, class TInitFunc>
        inline void AddSmartPtrImpl(TInitFunc initFunc, IBinSaver* binSaver, TPtr* ptr) {
            if (binSaver->IsReading()) {
                bool nonEmpty = false;
                binSaver->Add(0, &nonEmpty);
                if (nonEmpty) {
                    *ptr = initFunc();
                    binSaver->Add(0, ptr->Get());
                } else {
                    *ptr = nullptr;
                }
            } else {
                bool nonEmpty = ptr->Get() != nullptr;
                binSaver->Add(0, &nonEmpty);
                if (nonEmpty) {
                    binSaver->Add(0, ptr->Get());
                }
            }

        }

    }


    template <class T>
    inline void AddWithShared(IBinSaver* binSaver, TIntrusivePtr<T>* ptr) {
        NPrivate::AddSmartPtrImpl(MakeIntrusive<T>, binSaver, ptr);
    }

    template <class T>
    inline void AddWithShared(IBinSaver* binSaver, TVector<TIntrusivePtr<T>>* data) {
        IBinSaver::TStoredSize size = 0;
        if (binSaver->IsReading()) {
            data->clear();
            binSaver->Add(0, &size);
            data->resize(size);
        } else {
            size = SafeIntegerCast<IBinSaver::TStoredSize>(data->size());
            binSaver->Add(0, &size);
        }
        for (auto i : xrange(size)) {
            AddWithShared(binSaver, &((*data)[i]));
        }
    }

    template <class T>
    inline void AddWithShared(IBinSaver* binSaver, TAtomicSharedPtr<T>* ptr) {
        NPrivate::AddSmartPtrImpl(MakeAtomicShared<T>, binSaver, ptr);
    }

    template <class TKey, class TAnySharedPtr>
    inline void AddWithShared(IBinSaver* binSaver, THashMap<TKey, TAnySharedPtr>* data) {
        IBinSaver::TStoredSize size = 0;
        if (binSaver->IsReading()) {
            data->clear();
            LoadMulti(binSaver, &size);
            for (auto i : xrange(size)) {
                Y_UNUSED(i);
                TKey key;
                LoadMulti(binSaver, &key);
                TAnySharedPtr value;
                AddWithShared(binSaver, &value);
                data->emplace(std::move(key), std::move(value));
            }
        } else {
            size = SafeIntegerCast<IBinSaver::TStoredSize>(data->size());
            SaveMulti(binSaver, size);
            for (auto& keyValue : *data) { // not const because AddWithShared has non-const argument
                SaveMulti(binSaver, keyValue.first);
                AddWithShared(binSaver, &keyValue.second);
            }
        }
    }

    template <typename TOne>
    inline void AddWithSharedMulti(IBinSaver* binSaver, TOne& one) {
        AddWithShared(binSaver, &one);
    }

    template <typename THead, typename... TTail>
    inline void AddWithSharedMulti(IBinSaver* binSaver, THead& head, TTail&... tail) {
        AddWithShared(binSaver, &head);
        AddWithSharedMulti(binSaver, tail...);
    }

#define SAVELOAD_WITH_SHARED(...) \
    inline int operator&(IBinSaver& f) { \
        NCB::AddWithSharedMulti(&f, __VA_ARGS__); \
        return 0; \
    }

    void AddPadding(TCountingOutput* const output, ui32 alignment);
    void SkipPadding(TCountingInput* const input, ui32 alignment);

    void WriteMagic(const char* magic, ui32 magicSize, ui32 alignment, IOutputStream* stream);
    void ReadMagic(const char* expectedMagic, ui32 magicSize, ui32 alignment, IInputStream* stream);
}

template <class T>
inline int operator&(THolder<T>& ptr, IBinSaver& binSaver) {
    NCB::NPrivate::AddSmartPtrImpl(MakeHolder<T>, &binSaver, &ptr);
    return 0;
}


int operator&(NJson::TJsonValue& jsonValue, IBinSaver& binSaver);
