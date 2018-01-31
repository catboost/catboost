#pragma once

#include <catboost/libs/helpers/exception.h>

#include <util/generic/vector.h>
#include <util/system/yassert.h>
#include <util/ysaveload.h>

struct TSlice {
    ui64 Left;  //inclusive
    ui64 Right; //exclusive

    TSlice(ui64 left)
        : Left(left)
        , Right(left + 1)
    {
    }

    TSlice()
        : Left(0)
        , Right(0)
    {
    }

    ui64 Size() const {
        Y_ASSERT(Left <= Right);
        return Right - Left;
    }

    TSlice& operator+=(ui64 step) {
        Left += step;
        Right += step;
        return *this;
    }

    TSlice& operator*=(ui64 repeat) {
        ui64 size = Size();
        Left += repeat * size;
        Right += repeat * size;
        return *this;
    }

    TSlice(ui64 left,
           ui64 right)
        : Left(left)
        , Right(right)
    {
        Y_ASSERT(left <= right);
    }

    bool IsEmpty() const {
        return Left >= Right;
    }

    bool NotEmpty() const {
        return !IsEmpty();
    }

    static TSlice Intersection(const TSlice& lhs, const TSlice& rhs) {
        TSlice intersection(0, 0);
        intersection.Left = std::max(lhs.Left, rhs.Left);
        intersection.Right = std::min(lhs.Right, rhs.Right);
        if (intersection.Left >= intersection.Right) {
            intersection.Left = intersection.Right = 0;
        }
        return intersection;
    }

    bool Contains(const TSlice& slice) const {
        return slice.IsEmpty() || (Left <= slice.Left && slice.Right <= Right);
    }

    static TVector<TSlice> Remove(const TSlice& from, const TSlice& slice) {
        CB_ENSURE(from.Contains(slice));

        TVector<TSlice> result;
        if (slice.IsEmpty()) {
            result.push_back(from);
            return result;
        }

        if (slice.Left > from.Left) {
            result.push_back(TSlice(from.Left, slice.Left));
        }
        if (slice.Right < from.Right) {
            result.push_back(TSlice(slice.Right, from.Right));
        }

        return result;
    }

    bool operator<(const TSlice& other) const {
        return std::tie(Left, Right) < std::tie(other.Left, other.Right);
    }

    Y_SAVELOAD_DEFINE(Left, Right);
};

inline TSlice operator==(const TSlice& lhs, ui64 offset) {
    return TSlice(lhs.Left + offset, lhs.Right + offset);
}

inline bool operator==(const TSlice& lhs, const TSlice& rhs) {
    return (lhs.IsEmpty() && rhs.IsEmpty()) || (lhs.Left == rhs.Left && lhs.Right == rhs.Right);
}
