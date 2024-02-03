#include "util.h"

#include <catboost/private/libs/index_range/index_range.h>

#include <catboost/libs/data/target.h>
#include <catboost/libs/helpers/exception.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/is_in.h>
#include <util/generic/ymath.h>
#include <util/string/cast.h>
#include <util/string/escape.h>
#include <util/system/guard.h>
#include <util/system/mutex.h>

#include <variant>


using namespace NCB;


TTargetStats NCB::ComputeTargetStatsForYetiRank(
    const TRawTargetDataProvider& rawTargetData,
    ELossFunction lossFunction,
    NPar::ILocalExecutor* localExecutor
) {
    CB_ENSURE_INTERNAL(
        IsIn({ELossFunction::YetiRank, ELossFunction::YetiRankPairwise},lossFunction),
        "ComputeTargetStatsForYetiRank is intended to be used only with YetiRank or YetiRankPairwise"
    );

    const auto maybeTarget = rawTargetData.GetTarget();
    CB_ENSURE(maybeTarget, lossFunction << " loss function requires target data");
    CB_ENSURE(
        maybeTarget->size() == 1,
        lossFunction << " loss function requires single-dimensional target data"
    );

    auto processInBlocks = [&] (size_t size, std::function<void(TIndexRange<int>)> onBlock) {
        auto sizeAsInt = SafeIntegerCast<int>(size);
        TSimpleIndexRangesGenerator<int> indexRangeGenerator(
            TIndexRange<int>(sizeAsInt),
            Max(10000, CeilDiv(sizeAsInt, localExecutor->GetThreadCount()))
        );
        localExecutor->ExecRangeWithThrow(
            [&] (int blockIdx) {
                onBlock(indexRangeGenerator.GetRange(blockIdx));
            },
            0,
            indexRangeGenerator.RangesCount(),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    };


    TTargetStats result;
    TMutex resultMutex;

    const TRawTarget& rawTarget = maybeTarget->front();
    if (const ITypedSequencePtr<float>* floatSequence = std::get_if<ITypedSequencePtr<float>>(&rawTarget)) {
        processInBlocks(
            floatSequence->Get()->GetSize(),
            [&] (TIndexRange<int> block) {
                TTargetStats targetStats;

                auto blockIterator = floatSequence->Get()->GetBlockIterator(
                    TIndexRange<ui32>(block.Begin, block.End)
                );
                for (auto value : blockIterator->NextExact(block.GetSize())) {
                    CB_ENSURE(
                        !IsNan(value),
                        lossFunction << " loss function does not support NaNs in target data"
                    );
                    targetStats.Update(value);
                }

                with_lock (resultMutex) {
                    result.Update(targetStats);
                }
            }
        );
    } else {
        TConstArrayRef<TString> stringLabels = std::get<TVector<TString>>(rawTarget);

        processInBlocks(
            stringLabels.size(),
            [&, stringLabels] (TIndexRange<int> block) {
                TTargetStats targetStats;

                for (auto i : block.Iter()) {
                    const auto& stringLabel = stringLabels[i];
                    float floatLabel;
                    CB_ENSURE(
                        TryFromString(stringLabel, floatLabel),
                        "Target value \"" << EscapeC(stringLabel) << "\" cannot be parsed as float"
                    );
                    targetStats.Update(floatLabel);
                }

                with_lock (resultMutex) {
                    result.Update(targetStats);
                }
            }
        );
    }

    return result;
}
