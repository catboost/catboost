#pragma once

#include "order.h"
#include "util.h"

#include <catboost/private/libs/data_types/query.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/element_range.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

#include <utility>


namespace NCB {

    void CheckIsConsecutive(TConstArrayRef<TGroupBounds> groups);


    class TObjectsGrouping : public TThrRefBase {
    public:
        // needed because of default init in Cython and because of BinSaver
        TObjectsGrouping() = default;

        explicit TObjectsGrouping(ui32 groupCount) noexcept // trivial, all objects are groups of size 1
            : GroupCount(groupCount)
        {}

        explicit TObjectsGrouping(TVector<TGroupBounds>&& groups, bool skipCheck = false)
            : GroupCount(SafeIntegerCast<ui32>(groups.size()))
        {
            if (!skipCheck) {
                CheckIsConsecutive(groups);
            }
            Groups = std::move(groups);
        }

        bool operator==(const TObjectsGrouping& rhs) const noexcept {
            if (IsTrivial()) {
                if (rhs.IsTrivial()) {
                    return GroupCount == rhs.GroupCount;
                }
                return (GroupCount == rhs.GroupCount) &&
                    !FindIfPtr(
                        rhs.Groups,
                        [](TGroupBounds groupBounds) {
                            return groupBounds.GetSize() != 1;
                        }
                    );
            }
            return Groups == rhs.Groups;
        }

        SAVELOAD(GroupCount, Groups);

        ui32 GetObjectCount() const {
            return IsTrivial() ? GroupCount : Groups.back().End;
        }

        ui32 GetGroupCount() const noexcept {
            return GroupCount;
        }

        bool IsTrivial() const noexcept {
            return Groups.empty();
        }

        TGroupBounds GetGroup(ui32 groupIdx) const {
            CB_ENSURE(
                groupIdx < GroupCount,
                "group index (" << groupIdx << ") is greater than groups count (" << GroupCount << ')'
            );
            if (IsTrivial()) {
                // treat non-grouped data as groups of size 1
                return TGroupBounds(groupIdx, groupIdx + 1);
            }
            return Groups[groupIdx];
        }

        /* for more effective implementation, when checks in each GetGroup are too expensive
         * (but check IsTrivial() first!)
         */
        TConstArrayRef<TGroupBounds> GetNonTrivialGroups() const {
            CB_ENSURE(!IsTrivial(), "Groups are trivial");
            return Groups;
        }

        ui32 GetGroupIdxForObject(ui32 objectIdx) const {
            CB_ENSURE(
                objectIdx < GetObjectCount(),
                "object index (" << objectIdx << ") is greater than object count (" << GetObjectCount() << ')'
            );
            if (IsTrivial()) {
                return objectIdx;
            }
            auto groupsIt = LowerBound(
                Groups.begin(),
                Groups.end(),
                objectIdx,
                [](TGroupBounds groupBounds, ui32 objectIdx) {
                    return groupBounds.End <= objectIdx;
                }
            );
            Y_ASSERT(groupsIt != Groups.end());
            return ui32(groupsIt - Groups.begin());
        }

    private:
        ui32 GroupCount;
        TVector<TGroupBounds> Groups;
    };

    using TObjectsGroupingPtr = TIntrusivePtr<TObjectsGrouping>;

    class TObjectsGroupingSubset {
    public:
        // needed because of default init in Cython
        TObjectsGroupingSubset() = default;

        TObjectsGroupingSubset(
            TObjectsGroupingPtr subsetGrouping,
            TArraySubsetIndexing<ui32>&& groupsSubset,
            EObjectsOrder groupSubsetOrder,
            TMaybe<TArraySubsetIndexing<ui32>>&& objectsSubsetForNonTrivialGrouping
                = TMaybe<TArraySubsetIndexing<ui32>>(),

            // used only if objectsSubsetForNonTrivialGrouping is specified
            EObjectsOrder objectSubsetOrder = EObjectsOrder::Undefined
        )
            : SubsetGrouping(std::move(subsetGrouping))
            , GroupsSubset(std::move(groupsSubset))
            , GroupSubsetOrder(groupSubsetOrder)
            , ObjectsSubsetForNonTrivialGrouping(std::move(objectsSubsetForNonTrivialGrouping))
            , ObjectSubsetOrder(objectSubsetOrder)
        {
            CB_ENSURE(SubsetGrouping, "subsetGrouping must be initialized");
        }

        bool operator==(const TObjectsGroupingSubset& rhs) const {
            if (!(*SubsetGrouping == *rhs.SubsetGrouping) ||
                (GroupsSubset != rhs.GroupsSubset))
            {
                return false;
            }
            if (ObjectsSubsetForNonTrivialGrouping) {
                if (rhs.ObjectsSubsetForNonTrivialGrouping) {
                    return *ObjectsSubsetForNonTrivialGrouping == *rhs.ObjectsSubsetForNonTrivialGrouping;
                }
                return false;
            }
            return !rhs.ObjectsSubsetForNonTrivialGrouping;
        }

        TObjectsGroupingPtr GetSubsetGrouping() const {
            return SubsetGrouping;
        }

        const TArraySubsetIndexing<ui32>& GetGroupsIndexing() const {
            return GroupsSubset;
        }

        const TArraySubsetIndexing<ui32>& GetObjectsIndexing() const {
            return ObjectsSubsetForNonTrivialGrouping ?
                *ObjectsSubsetForNonTrivialGrouping : GroupsSubset;
        }

        EObjectsOrder GetGroupSubsetOrder() const {
            return GroupSubsetOrder;
        }

        EObjectsOrder GetObjectSubsetOrder() const {
            return ObjectsSubsetForNonTrivialGrouping ? GroupSubsetOrder : ObjectSubsetOrder;
        }

    private:
        TObjectsGroupingPtr SubsetGrouping = nullptr;

        TArraySubsetIndexing<ui32> GroupsSubset;
        EObjectsOrder GroupSubsetOrder = EObjectsOrder::Undefined;

        // defined only if grouping is non-trivial
        TMaybe<TArraySubsetIndexing<ui32>> ObjectsSubsetForNonTrivialGrouping;
        // used only if ObjectsSubsetForNonTrivialGrouping is defined
        EObjectsOrder ObjectSubsetOrder = EObjectsOrder::Undefined;
    };

    TObjectsGroupingSubset GetSubset(
        TObjectsGroupingPtr objectsGrouping,
        TArraySubsetIndexing<ui32>&& groupsSubset,
        EObjectsOrder groupSubsetOrder
    );

    TObjectsGroupingSubset GetGroupingSubsetFromObjectsSubset(
        TObjectsGroupingPtr objectsGrouping,
        TArraySubsetIndexing<ui32>&& objectsSubset,
        EObjectsOrder subsetOrder
    );

    // simplified interface for Cython
    inline TObjectsGroupingSubset GetGroupingSubsetFromObjectsSubset(
        TObjectsGroupingPtr objectsGrouping,
        TIndexedSubset<ui32>& objectsSubset,
        EObjectsOrder subsetOrder
    ) {
        return GetGroupingSubsetFromObjectsSubset(
            objectsGrouping,
            TArraySubsetIndexing<ui32>(std::move(objectsSubset)),
            subsetOrder
        );
    }


    TObjectsGroupingSubset Shuffle(
        TObjectsGroupingPtr objectsGrouping,
        ui32 permuteBlockSize,
        TRestorableFastRng64* rand
    );

    // returns groups (possibly trivial groups) subsets
    TVector<TArraySubsetIndexing<ui32>> Split(
        const TObjectsGrouping& objectsGrouping,
        ui32 partCount,
        bool oldCvStyle = false
    );

    void TrainTestSplit(
        const TObjectsGrouping& objectsGrouping,
        double trainPart,
        TArraySubsetIndexing<ui32>* trainIndices,
        TArraySubsetIndexing<ui32>* testIndices
    );

    static inline ui32 GetClassSplitMinLen(
        ui32 objectCount,
        const TVector<TVector<ui32>>& splittedByClass
    ) {
        ui32 minLen = objectCount;
        for (const auto& part : splittedByClass) {
            if (part.size() < (size_t)minLen) {
                minLen = (ui32)part.size();
            }
        }
        return minLen;
    }

    template <class TClassId>
    TVector<TVector<ui32>> SplitByClass(
        const TObjectsGrouping& objectsGrouping,
        TConstArrayRef<TClassId> objectClasses
    ) {
        CB_ENSURE(objectsGrouping.IsTrivial(), "Stratified split is not supported for data with groups");

        const ui32 objectCount = objectsGrouping.GetObjectCount();

        CheckDataSize(
            objectClasses.size(),
            (size_t)objectCount,
            "objectClasses",
            false,
            "objects size",
            true
        );

        TVector<std::pair<TClassId, ui32>> classWithObject;
        classWithObject.reserve(objectCount);
        for (ui32 i = 0; i < objectCount; ++i) {
            classWithObject.emplace_back(objectClasses[i], i);
        }
        Sort(classWithObject.begin(), classWithObject.end());

        TVector<TVector<ui32>> splittedByClass;
        for (ui32 i = 0; i < classWithObject.size(); ++i) {
            if (i == 0 || classWithObject[i].first != classWithObject[i - 1].first) {
                splittedByClass.emplace_back();
            }
            splittedByClass.back().push_back(classWithObject[i].second);
        }

        return splittedByClass;
    }

    template <class TClassId>
    void StratifiedTrainTestSplit(
        const TObjectsGrouping& objectsGrouping,
        TConstArrayRef<TClassId> objectClasses,
        double trainPart,
        TArraySubsetIndexing<ui32>* trainIndices,
        TArraySubsetIndexing<ui32>* testIndices
    ) {
        TVector<TVector<ui32>> splittedByClass = SplitByClass(objectsGrouping, objectClasses);
        ui32 minLen = GetClassSplitMinLen(objectsGrouping.GetObjectCount(), splittedByClass);
        if (minLen < 2) {
            CATBOOST_WARNING_LOG << " Warning: The least populated class in y has only "
                << minLen << " members, which is too few.";
        }
        TVector<ui32> resultTrainIndices;
        TVector<ui32> resultTestIndices;
        for (const auto& part : splittedByClass) {
            for (ui32 idx = 0; idx < part.size() * trainPart; ++idx) {
                resultTrainIndices.push_back(part[idx]);
            }
            for (ui32 idx = part.size() * trainPart; idx < part.size(); ++idx) {
                resultTestIndices.push_back(part[idx]);
            }
        }

        CB_ENSURE(!resultTrainIndices.empty(), "Not enough objects for splitting into train and test subsets");
        CB_ENSURE(!resultTestIndices.empty(), "Not enough objects for splitting into train and test subsets");

        Sort(resultTrainIndices.begin(), resultTrainIndices.end());
        *trainIndices = TArraySubsetIndexing<ui32>(std::move(resultTrainIndices));

        Sort(resultTestIndices.begin(), resultTestIndices.end());
        *testIndices = TArraySubsetIndexing<ui32>(std::move(resultTestIndices));
    }

    template <class TClassId>
    TVector<TArraySubsetIndexing<ui32>> StratifiedSplitToFolds(
        const TObjectsGrouping& objectsGrouping,
        TConstArrayRef<TClassId> objectClasses,
        ui32 partCount
    ) {
        TVector<TVector<ui32>> splittedByClass = SplitByClass(objectsGrouping, objectClasses);
        ui32 minLen = GetClassSplitMinLen(objectsGrouping.GetObjectCount(), splittedByClass);

        if (minLen < partCount) {
            CATBOOST_WARNING_LOG << " Warning: The least populated class in y has only "
                << minLen << " members, which is too few."
                " The minimum number of members in any class cannot be less than parts count="
                << partCount << Endl;
        }

        TVector<TVector<ui32>> resultIndices(partCount);
        for (const auto& part : splittedByClass) {
            for (ui32 fold = 0; fold < partCount; ++fold) {
                ui32 foldStartIndex, foldEndIndex;
                InitElementRange(fold, partCount, part.size(), &foldStartIndex, &foldEndIndex);
                for (ui32 idx = foldStartIndex; idx < foldEndIndex; ++idx) {
                    resultIndices[fold].push_back(part[idx]);
                }
            }
        }

        TVector<TArraySubsetIndexing<ui32>> result;
        for (auto& part : resultIndices) {
            CB_ENSURE(!part.empty(), "Not enough objects for splitting into " << partCount << " parts");
            Sort(part.begin(), part.end());
            result.push_back(TArraySubsetIndexing<ui32>(std::move(part)));
        }
        return result;
    }

    TVector<TArraySubsetIndexing<ui32>> SplitByObjects(
        const TObjectsGrouping& objectsGrouping,
        ui32 partSizeInObjects
    );

    TVector<TArraySubsetIndexing<ui32>> SplitByGroups(
        const TObjectsGrouping& objectsGrouping,
        ui32 partSizeInGroups
    );

    TVector<TArraySubsetIndexing<ui32>> QuantileSplitByObjects(
        const TObjectsGrouping& objectsGrouping,
        TConstArrayRef<ui64> timestamps,
        ui64 timesplitQuantileTimestamp,
        ui32 learnPartSizeInObjects
    );

    TVector<TArraySubsetIndexing<ui32>> QuantileSplitByGroups(
        const TObjectsGrouping& objectsGrouping,
        TConstArrayRef<ui64> timestamps,
        ui64 timesplitQuantileTimestamp,
        ui32 learnPartSizeInGroups
    );

    using TTimeSeriesTrainTestSubsets = std::pair<TVector<TArraySubsetIndexing<ui32>>,
                                                  TVector<TArraySubsetIndexing<ui32>>>;

    TTimeSeriesTrainTestSubsets TimeSeriesSplit(
        const TObjectsGrouping& objectsGrouping,
        ui32 partCount,
        bool oldCvStyle
    );
}
