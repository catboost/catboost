#pragma once

#include "split.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/hash.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/model/split.h>

#include <util/ysaveload.h>
#include <util/generic/hash_set.h>
#include <util/system/types.h>

#include <tuple>

namespace NMonoForest {
    struct TMonomStat {
        TVector<double> Value;
        double Weight = -1;

        double Norm() const {
            long double norm = 0;
            for (auto value : Value) {
                norm += value * value;
            }
            return norm;
        }

        bool operator==(const TMonomStat& rhs) const {
            return std::tie(Value, Weight) == std::tie(rhs.Value, rhs.Weight);
        }

        bool operator!=(const TMonomStat& rhs) const {
            return !(rhs == *this);
        }

        Y_SAVELOAD_DEFINE(Value, Weight);
    };

    struct TMonomStructure {
        TVector<TBinarySplit> Splits;

        ui32 GetDepth() const {
            return Splits.size();
        }

        void AddSplit(const TBinarySplit& split) {
            Splits.push_back(split);
        }

        bool operator==(const TMonomStructure& rhs) const {
            return std::tie(Splits) == std::tie(rhs.Splits);
        }

        bool operator!=(const TMonomStructure& rhs) const {
            return !(rhs == *this);
        }

        ui64 GetHash() const {
            return VecCityHash(Splits);
        }

        Y_SAVELOAD_DEFINE(Splits);
    };

    struct TMonom {
        TMonomStructure Structure;
        TMonomStat Stat;

        bool operator==(const TMonom& rhs) const {
            return std::tie(Structure, Stat) == std::tie(rhs.Structure, rhs.Stat);
        }

        bool operator!=(const TMonom& rhs) const {
            return !(rhs == *this);
        }

        Y_SAVELOAD_DEFINE(Structure, Stat);
    };
}

template <>
struct THash<NMonoForest::TMonomStructure> {
    inline size_t operator()(const NMonoForest::TMonomStructure& structure) const {
        return structure.GetHash();
    }
};
