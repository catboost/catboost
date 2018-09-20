#pragma once

#include <catboost/libs/data_types/query.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NCB {

    void CheckIsConsecutive(TConstArrayRef<TGroupBounds> groups);


    class TObjectsGrouping : public TThrRefBase {
    public:
        explicit TObjectsGrouping(ui32 groupCount) // trivial, all objects are groups of size 1
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

        ui32 GetObjectCount() const {
            return IsTrivial() ? GroupCount : Groups.back().End;
        }

        ui32 GetGroupCount() const {
            return GroupCount;
        }

        bool IsTrivial() const {
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

        bool operator==(const TObjectsGrouping& rhs) const {
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

    private:
        ui32 GroupCount;
        TVector<TGroupBounds> Groups;
    };

    using TObjectsGroupingPtr = TIntrusivePtr<TObjectsGrouping>;

    class TObjectsGroupingSubset {
    public:
        TObjectsGroupingSubset(
            TObjectsGroupingPtr subsetGrouping,
            TArraySubsetIndexing<ui32>&& groupsSubset,
            THolder<TArraySubsetIndexing<ui32>>&& objectsSubsetForNonTrivialGrouping
                = THolder<TArraySubsetIndexing<ui32>>()
        )
            : SubsetGrouping(std::move(subsetGrouping))
            , GroupsSubset(std::move(groupsSubset))
            , ObjectsSubsetForNonTrivialGrouping(std::move(objectsSubsetForNonTrivialGrouping))
        {
            CB_ENSURE(SubsetGrouping, "subsetGrouping must be initialized");
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

    private:
        TObjectsGroupingPtr SubsetGrouping;

        TArraySubsetIndexing<ui32> GroupsSubset;

        // created only if grouping is non-trivial
        THolder<TArraySubsetIndexing<ui32>> ObjectsSubsetForNonTrivialGrouping;
    };

    TObjectsGroupingSubset GetSubset(
        TObjectsGroupingPtr objectsGrouping,
        TArraySubsetIndexing<ui32>&& groupsSubset
    );

}
