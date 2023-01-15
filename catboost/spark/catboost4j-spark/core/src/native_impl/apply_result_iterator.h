#pragma once

#include <catboost/libs/data/objects.h>

#include <catboost/private/libs/options/enums.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>
#include <util/system/yassert.h>


class TApplyResultIterator {
public:
    TApplyResultIterator(
        const TFullModel& model,
        NCB::TRawObjectsDataProviderPtr rawObjectsDataProvider,
        EPredictionType predictionType,
        i32 threadCount
    ) throw(yexception);

    double GetSingleDimensionalResult(i32 objectIdx) const {
        return ApplyResult[0][objectIdx];
    }

    const TVector<double>& GetSingleDimensionalResults() const {
        return ApplyResult[0];
    }

    void GetMultiDimensionalResult(i32 objectIdx, TArrayRef<double> result) const {
        if (ApplyResult.size() == 1) {
            Y_ASSERT(result.size() == 2);
            result[0] = (-0.5) * ApplyResult[0][objectIdx];
            result[1] = -result[0];
        } else {
            for (auto dimIdx : xrange(result.size())) {
                result[dimIdx] = ApplyResult[dimIdx][objectIdx];
            }
        }
    }

private:
    TVector<TVector<double>> ApplyResult; // [modelDimensionIdx][objectIdx]
};
