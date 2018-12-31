#pragma once

#include "feature.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/hash.h>

#include <util/digest/multi.h>
#include <util/generic/vector.h>
#include <util/system/types.h>
#include <util/str_stl.h>
#include <util/ysaveload.h>

#include <tuple>


namespace NCatboostCuda {


    struct TLeafPath {
        TVector<TBinarySplit> Splits;
        TVector<ESplitValue> Directions;

        ui32 GetDepth() const {
            return Splits.size();
        }

        void AddSplit(const TBinarySplit& split, ESplitValue direction) {
            Splits.push_back(split);
            Directions.push_back(direction);
        }

        bool operator==(const TLeafPath& rhs) const {
            return std::tie(Splits, Directions) == std::tie(rhs.Splits, rhs.Directions);
        }

        bool operator!=(const TLeafPath& rhs) const {
            return !(rhs == *this);
        }

        ui64 GetHash() const {
            return MultiHash(VecCityHash(Splits), VecCityHash(Directions));
        }

        Y_SAVELOAD_DEFINE(Splits, Directions);

    };

    inline TLeafPath PreviousSplit(const TLeafPath& path) {
        const size_t size = path.GetDepth();
        CB_ENSURE(size > 0, "Error: can't remove split");
        TLeafPath prevPath = path;
        prevPath.Splits.resize(size - 1);
        prevPath.Directions.resize(size - 1);
        return prevPath;
    }
}


template <>
struct THash<NCatboostCuda::TLeafPath> {
    inline size_t operator()(const NCatboostCuda::TLeafPath& path) const {
        return path.GetHash();
    }
};
