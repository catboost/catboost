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

        bool IsSorted() const {
            for (ui32 i = 1; i < Splits.size(); ++i) {
                if (Splits[i] <= Splits[i - 1]) {
                    return false;
                }
            }
            return true;
        }

        bool HasDuplicates() const {
            for (ui32 i = 1; i < Splits.size(); ++i) {
                if (Splits[i] == Splits[i - 1]) {
                    return true;
                }
            }
            return false;
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

    template <class TSortBy>
    inline TLeafPath SortPath(const TLeafPath& path, TSortBy&& cmpFunc) {
        TVector<ui32> indices(path.Splits.size());
        Iota(indices.begin(), indices.end(), 0);
        Sort(indices.begin(), indices.end(), [&](const ui32 left, const ui32 right) -> bool {
            return cmpFunc(path.Splits[left], path.Splits[right]);
        });
        auto newPath = path;
        for (ui64 i = 0; i < indices.size(); ++i) {
            const ui32 loadIdx = indices[i];
            newPath.Splits[i] = path.Splits[loadIdx];
            newPath.Directions[i] = path.Directions[loadIdx];
        }
        return newPath;
    }

    template <class TSortBy>
    inline TLeafPath SortUniquePath(const TLeafPath& path, TSortBy&& cmpFunc) {
        TVector<ui32> indices(path.Splits.size());
        Iota(indices.begin(), indices.end(), 0);
        Sort(indices, [&](const ui32 left, const ui32 right) -> bool {
            return cmpFunc(path.Splits[left], path.Splits[right]);
        });

        auto last = std::unique(indices.begin(), indices.end(), [&](const ui32 left, const ui32 right) -> bool {
            return path.Splits[left] == path.Splits[right] && path.Directions[left] == path.Directions[right];
        });
        indices.resize(last - indices.begin());

        TLeafPath newPath;
        newPath.Splits.resize(indices.size());
        newPath.Directions.resize(indices.size());

        for (ui64 i = 0; i < indices.size(); ++i) {
            const ui32 loadIdx = indices[i];
            newPath.Splits[i] = path.Splits[loadIdx];
            newPath.Directions[i] = path.Directions[loadIdx];
        }
        CB_ENSURE(!newPath.HasDuplicates());
        return newPath;
    }
}

template <>
struct THash<NCatboostCuda::TLeafPath> {
    inline size_t operator()(const NCatboostCuda::TLeafPath& path) const {
        return path.GetHash();
    }
};
