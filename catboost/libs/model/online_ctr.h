#pragma once

#include "projection.h"
#include <catboost/libs/ctr_description/ctr_type.h>
#include <library/containers/2d_array/2d_array.h>
#include <library/containers/dense_hash/dense_hash.h>
#include <util/ysaveload.h>

struct TModelCtr {
    TProjection Projection;
    ECtrType CtrType = ECtrType::Borders;
    int TargetBorderClassifierIdx = 0;
    int TargetBorderIdx = 0;
    float PriorNum = 0.0f;
    float PriorDenom = 1.0f;
    float Shift = 0.0f;
    float Scale = 1.0f;

    Y_SAVELOAD_DEFINE(Projection, CtrType, TargetBorderClassifierIdx, TargetBorderIdx, PriorNum, PriorDenom, Shift, Scale);

    TModelCtr() = default;

    bool operator==(const TModelCtr& other) const {
        return std::tie(Projection, CtrType, TargetBorderClassifierIdx, TargetBorderIdx, PriorNum, PriorDenom, Shift, Scale) ==
               std::tie(other.Projection, other.CtrType, other.TargetBorderClassifierIdx, other.TargetBorderIdx, other.PriorNum, other.PriorDenom, other.Shift, other.Scale);
    }

    bool operator!=(const TModelCtr& other) const {
        return !(*this == other);
    }

    bool operator<(const TModelCtr& other) const {
        return std::tie(Projection, CtrType, TargetBorderClassifierIdx, TargetBorderIdx, PriorNum, PriorDenom, Shift, Scale) <
               std::tie(other.Projection, other.CtrType, other.TargetBorderClassifierIdx, other.TargetBorderIdx, other.PriorNum, other.PriorDenom, other.Shift, other.Scale);
    }

    size_t GetHash() const {
        return MultiHash(Projection.GetHash(), CtrType, TargetBorderClassifierIdx, TargetBorderIdx, PriorNum, PriorDenom, Shift, Scale);
    }

    inline float Calc(float countInClass, float totalCount) const {
        float ctr = (countInClass + PriorNum) / (totalCount + PriorDenom);
        return (ctr + Shift) * Scale;
    }
};

template <>
struct THash<TModelCtr> {
    size_t operator()(const TModelCtr& ctr) const noexcept {
        return ctr.GetHash();
    }
};

struct TModelCtrSplit {
    TModelCtr Ctr;
    float Border = 0.0f;

    TModelCtrSplit() = default;
    size_t GetHash() const {
        return MultiHash(Ctr, Border);
    }
    bool operator==(const TModelCtrSplit& other) const {
        return std::tie(Ctr, Border) == std::tie(other.Ctr, other.Border);
    }

    bool operator!=(const TModelCtrSplit& other) const {
        return !(*this == other);
    }

    bool operator<(const TModelCtrSplit& other) const {
        return std::tie(Ctr, Border) < std::tie(other.Ctr, other.Border);
    }
    Y_SAVELOAD_DEFINE(Ctr, Border);
};

template <>
struct THash<TModelCtrSplit> {
    size_t operator()(const TModelCtrSplit& ctr) const noexcept {
        return ctr.GetHash();
    }
};

struct TCtrHistory {
    int N[2];
    void Clear() {
        N[0] = 0;
        N[1] = 0;
    }
    Y_SAVELOAD_DEFINE(N[0], N[1]);
};

struct TCtrMeanHistory {
    float Sum;
    int Count;
    bool operator==(const TCtrMeanHistory& other) const {
        return std::tie(Sum, Count) == std::tie(other.Sum, other.Count);
    }
    void Clear() {
        Sum = 0;
        Count = 0;
    }
    void Add(float target) {
        Sum += target;
        ++Count;
    }
    Y_SAVELOAD_DEFINE(Sum, Count);
};
