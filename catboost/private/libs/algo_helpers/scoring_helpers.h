#pragma once

#include <catboost/private/libs/algo/calc_score_cache.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>


template <class T>
class TDataRefOptionalHolder {
public:
    TDataRefOptionalHolder() = default;

    // Buf not used, init from external data
    explicit TDataRefOptionalHolder(TArrayRef<T> extData)
        : Data(extData) {}

    // noninitializing
    explicit TDataRefOptionalHolder(size_t size) {
        Buf.yresize(size);
        Data = TArrayRef<T>(Buf);
    }

    bool NonInited() const {
        return Data.data() == nullptr;
    }

    TArrayRef<T> GetData() {
        return Data;
    }

    TConstArrayRef<T> GetData() const {
        return Data;
    }

private:
    TArrayRef<T> Data;
    TVector<T> Buf;
};

using TBucketStatsRefOptionalHolder = TDataRefOptionalHolder<TBucketStats>;

/* A helper function that returns calculated ctr values for this projection
   (== feature or feature combination) from cache.
*/
inline const TOnlineCtrBase& GetCtr(
    const std::tuple<const TOnlineCtrBase&, const TOnlineCtrBase&>& allCtrs,
    const TProjection& proj
) {
    static const constexpr size_t OnlineSingleCtrsIndex = 0;
    static const constexpr size_t OnlineCTRIndex = 1;
    return proj.HasSingleFeature() ? std::get<OnlineSingleCtrsIndex>(allCtrs)
                                   : std::get<OnlineCTRIndex>(allCtrs);
}
