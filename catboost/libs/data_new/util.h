#pragma once

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


    /* TODO(akhropov): uncomment when custom policy support is fixed in TMaybe implementation
     *
    struct TPolicyUnavailableData {
        static void OnEmpty() {
            CB_ENSURE_INTERNAL(false, "Data is unavailable");
        }
    };

    template <class T>
    using TMaybeData = TMaybe<T, TPolicyUnavailableData>;

    */

    template <class T>
    using TMaybeData = TMaybe<T>;

}
