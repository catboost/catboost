#pragma once

#include <catboost/private/libs/data_types/pair.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe.h>

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
        const TStringBuf expectedSizeName = "object count",
        bool internalCheck = false
    ) {
        CB_ENSURE(
            (dataCanBeEmpty && (dataSize == 0)) || (dataSize == expectedSize),
            (internalCheck ? NCB::INTERNAL_ERROR_MSG : TStringBuf()) << dataName << " data size ("
            << dataSize << ") is not equal to " << expectedSizeName << " (" << expectedSize << ')'
        );
    }

    template <class T>
    void PrepareForInitialization(
        size_t size,
        size_t prevTailSize,
        TVector<T>* data
    ) {
        if (prevTailSize) {
            CB_ENSURE(prevTailSize <= size, "Data remainder is too large");
            CB_ENSURE(prevTailSize <= data->size(), "Data remainder is too large");
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
                CB_ENSURE(prevTailSize == 0, "Data remainder should be empty");
                dataRef = TVector<T>();
            }
            PrepareForInitialization(size, prevTailSize, &*dataRef);
        } else {
            dataRef = Nothing();
        }
    }

    template <class T>
    void PrepareForInitialization(
        size_t dataCount,
        size_t size,
        size_t prevTailSize,
        TVector<TVector<T>>* data
    ) {
        data->resize(dataCount);
        for (auto& subData : *data) {
            PrepareForInitialization(size, prevTailSize, &subData);
        }
    }
}
