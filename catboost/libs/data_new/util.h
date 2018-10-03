#pragma once

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>

#include <util/system/types.h>


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
}
