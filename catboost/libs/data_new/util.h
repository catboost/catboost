#pragma once

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/array_ref.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>

#include <util/system/types.h>

#include <algorithm>


namespace NCB {

    template <class T>
    void CheckDataSize(
        T dataSize,
        T expectedSize,
        const TStringBuf dataName,
        bool dataCanBeEmpty = false,
        const TStringBuf expectedSizeName = AsStringBuf("object count"),
        bool internalCheck = false
    ) {
        CB_ENSURE(
            (dataCanBeEmpty && (dataSize == 0)) || (dataSize == expectedSize),
            (internalCheck ? NCB::INTERNAL_ERROR_MSG : TStringBuf()) << dataName << " data size ("
            << dataSize << ") is not equal to " << expectedSizeName << " (" << expectedSize << ')'
        );
    }


    struct TPolicyUnavailableData {
        static void OnEmpty() {
            CB_ENSURE_INTERNAL(false, "Attempt to access unavailable data");
        }
    };

    template <class T>
    using TMaybeData = TMaybe<T, TPolicyUnavailableData>;

    template <class T, class TPtr>
    TMaybeData<T*> MakeMaybeData(const TPtr& ptr) {
        return ptr ? TMaybeData<T*>(ptr.Get()) : Nothing();
    }

    template <class T>
    TMaybeData<T*> MakeMaybeData(T* ptr) {
        return ptr ? TMaybeData<T*>(ptr) : Nothing();
    }

    // pairs are a special case because there's no guaranteed order for now, so compare them as multisets
    bool EqualAsMultiSets(TConstArrayRef<TPair> lhs, TConstArrayRef<TPair> rhs);

    template <class T>
    void PrepareForInitialization(
        size_t size,
        size_t prevTailSize,
        TVector<T>* data
    ) {
        if (prevTailSize) {
            Y_VERIFY(prevTailSize <= size);
            Y_VERIFY(prevTailSize <= data->size());
            std::move(data->end() - prevTailSize, data->end(), data->begin());
        }
        data->yresize(size);
    }


    template <class T>
    void PrepareForInitialization(
        bool defined,
        size_t size,
        size_t prevTailSize,
        TMaybeData<TVector<T>>* data
    ) {
        auto& dataRef = *data;
        if (defined) {
            if (!dataRef) {
                Y_VERIFY(prevTailSize == 0);
                dataRef = TVector<T>();
            }
            PrepareForInitialization(size, prevTailSize, &*dataRef);
        } else {
            dataRef = Nothing();
        }
    }

}
