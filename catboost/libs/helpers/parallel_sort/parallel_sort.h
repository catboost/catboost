#pragma once

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>

namespace NCB {

    struct TMergeData {
        ui32 Left1;
        ui32 Right1;
        ui32 Left2;
        ui32 Right2;
        ui32 OutputIndex;

        ui32 GetSize() const {
            return Right1 - Left1 + Right2 - Left2;
        }
    };

    void EquallyDivide(ui32 size, ui32 blockCount, TVector<ui32>* blockSizes);

    template <class TElement, typename TCompare>
    inline void DivideMergeIntoParallelMerges(
        const TMergeData& mergeOperation,
        TCompare cmp,
        const TVector<TElement>& elements,
        TVector<TMergeData>* mergeData,
        ui32* threadsForMerge
    ) {
        ui32 left1 = mergeOperation.Left1;
        ui32 left2 = mergeOperation.Left2;
        ui32 outputIndex = mergeOperation.OutputIndex;
        const ui32 currentSize = mergeOperation.GetSize();
        const ui32 currentThreadsCount = Min(currentSize, *threadsForMerge);
        *threadsForMerge = currentThreadsCount;
        TVector<ui32> mergeSizes;
        EquallyDivide(currentSize, currentThreadsCount, &mergeSizes);
        for (ui32 j = 0; j + 1 < currentThreadsCount; ++j) {
            const ui32 size = mergeSizes[j];
            ui32 leftPosition = 0;
            ui32 rightPosition = Min(size, mergeOperation.Right1 - left1) + 1;
            while (leftPosition + 1 < rightPosition) {
                ui32 currentPosition = (leftPosition + rightPosition) / 2u;
                ui32 currentLeftIndex = left1 + currentPosition - 1;
                ui32 currentRightIndex = left2 + size - currentPosition;
                if (currentRightIndex < mergeOperation.Right2 && cmp(elements[currentRightIndex], elements[currentLeftIndex])) {
                    rightPosition = currentPosition;
                } else {
                    leftPosition = currentPosition;
                }
            }
            const ui32 leftSize = leftPosition;
            const ui32 rightSize = size - leftSize;
            mergeData->push_back({left1, left1 + leftSize, left2, left2 + rightSize, outputIndex});
            left1 += leftSize;
            left2 += rightSize;
            outputIndex += size;
        }
        mergeData->push_back({left1, mergeOperation.Right1, left2, mergeOperation.Right2, outputIndex});
    }

    template <class TElement, typename TCompare>
    inline void ParallelMergeSort(
        TCompare cmp,
        TVector<TElement>* elements,
        NPar::ILocalExecutor* localExecutor,
        TVector<TElement>* buf = nullptr
    ) {
        if (elements->size() <= 1u) {
            return;
        }
        TVector<TElement> newBuf;
        if (buf == nullptr) {
            newBuf.assign(elements->begin(), elements->end());
            buf = &newBuf;
        }
        const ui32 threadCount = Min((ui32)localExecutor->GetThreadCount() + 1, (ui32)elements->size());
        TVector<ui32> blockSizes;
        EquallyDivide(elements->size(), threadCount, &blockSizes);
        TVector<ui32> startPositions(threadCount);
        ui32 position = 0;
        for (ui32 i = 0; i < threadCount; ++i) {
            startPositions[i] = position;
            position += blockSizes[i];
        }
        NPar::ParallelFor(
            *localExecutor,
            0,
            threadCount,
            [&](int blockId) {
                int left = startPositions[blockId];
                int right = left + blockSizes[blockId];
                // used only in AUC
                Sort(elements->begin() + left, elements->begin() + right, cmp);
            }
        );
        while (blockSizes.size() > 1u) {
            const ui32 currentMergesCount = blockSizes.size() / 2u;
            TVector<ui32> threadsPerMergeCount;
            EquallyDivide(threadCount, currentMergesCount, &threadsPerMergeCount);
            TVector<TMergeData> mergeData;
            for (ui32 i = 0; i < currentMergesCount; ++i) {
                TMergeData currentMerge = {
                    startPositions[2 * i],
                    startPositions[2 * i + 1],
                    startPositions[2 * i + 1],
                    (2 * i + 2 == startPositions.size() ? static_cast<ui32>(elements->size()) : startPositions[2 * i + 2]),
                    startPositions[2 * i]
                };
                DivideMergeIntoParallelMerges(currentMerge, cmp, *elements, &mergeData, &threadsPerMergeCount[i]);
            }
            NPar::ParallelFor(
                *localExecutor,
                0,
                mergeData.size(),
                [&](int blockId) {
                    std::merge(
                        elements->begin() + mergeData[blockId].Left1,
                        elements->begin() + mergeData[blockId].Right1,
                        elements->begin() + mergeData[blockId].Left2,
                        elements->begin() + mergeData[blockId].Right2,
                        buf->begin() + mergeData[blockId].OutputIndex,
                        cmp
                    );
                }
            );
            NPar::ParallelFor(
                *localExecutor,
                0,
                mergeData.size(),
                [&](int blockId) {
                    int startPosition = mergeData[blockId].OutputIndex;
                    int endPosition = startPosition + mergeData[blockId].GetSize();
                    std::copy(buf->begin() + startPosition, buf->begin() + endPosition, elements->begin() + startPosition);
                }
            );
            const ui32 newSize = (blockSizes.size() + 1) / 2u;
            TVector<ui32> newBlockSizes, newStartPositions;
            newBlockSizes.reserve(newSize);
            newStartPositions.reserve(newSize);
            for (ui32 i = 0; i + 1 < blockSizes.size(); i += 2) {
                newBlockSizes.emplace_back(blockSizes[i] + blockSizes[i + 1]);
                newStartPositions.emplace_back(startPositions[i]);
            }
            if (2 * newSize != blockSizes.size()) {
                newBlockSizes.emplace_back(blockSizes.back());
                newStartPositions.emplace_back(startPositions.back());
            }
            blockSizes = newBlockSizes;
            startPositions = newStartPositions;
        }
    }
}
