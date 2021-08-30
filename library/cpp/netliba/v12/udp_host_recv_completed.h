#pragma once

#include <library/cpp/containers/disjoint_interval_tree/disjoint_interval_tree.h>
#include <util/datetime/base.h>
#include <util/generic/algorithm.h>
#include <util/generic/hash_set.h>
#include <util/generic/map.h>
#include <util/generic/typetraits.h>
#include <util/string/cast.h>
#include <util/system/defaults.h>
#include "circular_pod_buffer.h"

namespace NNetliba_v12 {
    template <class T>
    class THashSetWithMin {
    public:
        THashSetWithMin()
            : MinValue()
        {
        }

        void Insert(const T& t) {
            Set.insert(t);
            MinValue = Min(t, MinValue);
        }
        bool Has(const T& t) const {
            return Set.contains(t);
        }

        bool Empty() const {
            return Set.empty();
        }
        size_t Size() const {
            return Set.size();
        }

        void Clear() {
            Set.clear();
            MinValue = T();
        }
        void Swap(THashSetWithMin<T>& rhv) {
            std::swap(Set, rhv.Set);
            std::swap(MinValue, rhv.MinValue);
        }

        const T& GetMinValue() const {
            Y_ASSERT(!Set.empty());
            return MinValue;
        }

    private:
        THashSet<T> Set;
        T MinValue;
    };

    ///////////////////////////////////////////////////////////////////////////////

    // This code assumes that transferIds are mostly growing up sequence with elements differ by 1.
    //
    // We do not want to store all finished trasfers - with small transfers and large speed it will actually take megabytes!
    // So we store not-yet-started trasfers (holes). In most cases there will be no holes so we'll use little memory.
    // Worst case: client sends packets with large transfer id step: 1000 then 10000 then 15000. This may happen when
    // there are problems with network or server was restarted and client was retrying and retrying and some old packets received.
    // For fastest access we store information about last 1024 transfers in circular buffer and use interval trees for storing info
    // about old transfers. We need trees for a) fast intervals insertion b) fast min/max calculation.
    // We drop information about holes after some timeout.
    class TRecvCompleted {
    private:
        enum EState { ES_HOLE = 0,
                      ES_ACTIVE = 1,
                      ES_OK = 2,
                      ES_FAILED = 3,
                      ES_CANCELED = 4 };

    public:
        TRecvCompleted()
            : MinSeenId(0)
            , MaxSeenId(0)
            , Current(1024) // TODO: think about Current size
            , NumCurrentActive(0)
        {
        }

        bool IsCompleted(const ui64 id, bool* isFailed, bool* isCanceled) const {
            Y_ASSERT(id > 0);
            *isFailed = false;
            *isCanceled = false;

            // In average case OldHoles, OldActive and PrevHoles will be empty and all information is stored in Current.
            // OldHoles and PrevHoles may contain lots of elements but number of nodes in tree is expected to be small.
            // Number of elements in OldActive is expected to be near zero.

            if (MinSeenId <= id && id <= MaxSeenId) {
                if (IsInCurrent(id)) {
                    const char s = CurrentState(id);

                    if (s == ES_OK || s == ES_FAILED || s == ES_CANCELED) {
                        *isFailed = s == ES_FAILED;
                        *isCanceled = s == ES_CANCELED;
                        return true;
                    } else if (s == ES_HOLE || s == ES_ACTIVE) {
                        return false;
                    } else {
                        Y_ASSERT(false);
                    }
                } else if (OldFailed.Has(id) || PrevFailed.Has(id)) {
                    Y_ASSERT((int)OldFailed.Has(id) + (int)PrevFailed.Has(id) < 2);
                    *isFailed = true;
                    return true;
                } else if (OldCanceled.Has(id) || PrevCanceled.Has(id)) {
                    Y_ASSERT((int)OldCanceled.Has(id) + (int)PrevCanceled.Has(id) < 2);
                    *isCanceled = true;
                    return true;
                } else {
                    return !OldActive.Has(id) && !OldHoles.Has(id) && !PrevHoles.Has(id);
                }
            }
            return false;
        }

        void NewTransfer(const ui64 id) {
            Y_ASSERT(id > 0);

#ifndef NDEBUG
            bool dummy;
            Y_ASSERT(!IsCompleted(id, &dummy, &dummy)); // TODO: convert to VERIFY?
#endif

            if (id > MaxSeenId) {                 // most expected case
                if (Y_UNLIKELY(MaxSeenId == 0)) { // first transfer ever
                    Y_ASSERT(Current.Size() == 0);
                    MinSeenId = id;
                    MaxSeenId = id - 1; // will be incremented in PushBackToCurrent
                } else {
                    for (ui64 i = MaxSeenId + 1; i < id; ++i) { // TODO: optimize (batch)
                        PushBackToCurrent(ES_HOLE);
                    }
                }
                PushBackToCurrent(ES_ACTIVE);
                Y_ASSERT(MaxSeenId == id);
                return;
            }

            if (id < MinSeenId) {
                Y_ASSERT(MinSeenId > 0 && MaxSeenId > 0);
                if (id + 1 < MinSeenId) {
                    OldHoles.InsertInterval(id + 1, MinSeenId);
                }

                MinSeenId = id;
                CheckedInsert(&OldActive, id);
                return;
            }

            if (IsInCurrent(id)) {
                Y_ASSERT(CurrentState(id) == ES_HOLE);
                CurrentState(id) = ES_ACTIVE;
                NumCurrentActive++;

            } else {
                CheckedInsert(&OldActive, id);
                if (!OldHoles.Erase(id)) {
                    CheckedErase(&PrevHoles, id); // wow! such old transfer arrived
                } else {
                    Y_ASSERT(!PrevHoles.Has(id));
                }
            }
        }

        void MarkCompleted(const ui64 id, const bool isFailed, const bool isCanceled) {
            Y_ASSERT(id > 0);
            Y_ASSERT((int)isFailed + (int)isCanceled < 2); // but we also ensure that id will be inserted only once

            if (IsInCurrent(id)) {
                Y_ASSERT(CurrentState(id) == ES_ACTIVE);
                CurrentState(id) = isCanceled ? ES_CANCELED : (isFailed ? ES_FAILED : ES_OK);
                NumCurrentActive--;
            } else {
                CheckedErase(&OldActive, id);
                if (isCanceled) {
                    CheckedInsert(&OldCanceled, id);
                } else if (isFailed) {
                    CheckedInsert(&OldFailed, id);
                }
            }
        }

        bool Cleanup(const TGUID& guid) {
            const TInstant now = TInstant::Now();
            const TDuration timeout = TDuration::Minutes(15); // TODO: choose proper timeout

            if (now - PrevHolesCreationTime <= timeout || MaxSeenId == 0) {
                return false;
            }
            PrevHolesCreationTime = now;

            // this code tries to emulate original netliba semantics: forget guid of completed transfer after ~3 minutes
            PrevHoles.Swap(OldHoles);
            OldHoles.Clear();

            PrevCanceled.Swap(OldCanceled);
            OldCanceled.Clear();

            PrevFailed.Swap(OldFailed);
            OldFailed.Clear();

            MinSeenId = MaxSeenId - Current.Size();
            if (!PrevHoles.Empty()) {
                Y_ASSERT(PrevHoles.Min() - 1 < MinSeenId);
                MinSeenId = PrevHoles.Min() - 1; // PrevHoles.Min() is a hole - we have not seen such transfer, but we created it after prev transfer id
            }
            if (!OldActive.Empty()) {
                MinSeenId = Min(OldActive.Min(), MinSeenId);
            }
            if (!PrevCanceled.Empty()) {
                MinSeenId = Min(PrevCanceled.GetMinValue(), MinSeenId);
            }
            if (!PrevFailed.Empty()) {
                MinSeenId = Min(PrevFailed.GetMinValue(), MinSeenId);
            }

            /*
        const TDuration t = TInstant::Now() - now;
        printf("[%s] TRecvCompleted took: %dus, Min = %" PRIu64 ", Max = %" PRIu64 ", Current.Size = %d,"
               " PrevHoles.Elements = %d, PrevHoles.Intervals = %d, OldActive.Elements = %d, OldActive.Intervals = %d\n",
               GetGuidAsString(guid).c_str(), (int)t.MicroSeconds(), MinSeenId, MaxSeenId, (int)Current.Size(),
               (int)PrevHoles.GetNumElements(), (int)PrevHoles.GetNumIntervals(),
               (int)OldActive.GetNumElements(), (int)OldActive.GetNumIntervals());
        */
            Y_UNUSED(guid);
            return true;
        }

        void Clear() {
            *this = TRecvCompleted();
        }

        size_t GetNumActive() const {
            return NumCurrentActive + OldActive.GetNumElements();
        }

        TString GetDebugInfo() const {
            TString result;
            result += "min: ";
            result += ToString(MinSeenId);
            result += ", max: ";
            result += ToString(MaxSeenId);
            result += "; active: ";
            result += ToString(GetNumActive());
            result += " (current: ";
            result += ToString(NumCurrentActive);
            result += ", old: ";
            result += ToString(OldActive.GetNumElements());
            result += "); old holes: ";
            result += ToString(OldHoles.GetNumElements());
            result += " (";
            result += ToString(OldHoles.GetNumIntervals());
            result += "), prev holes: ";
            result += ToString(PrevHoles.GetNumElements());
            result += " (";
            result += ToString(PrevHoles.GetNumIntervals());
            result += ")";
            result += ", old canceled: ";
            result += ToString(OldCanceled.Size());
            result += ", prev canceled: ";
            result += ToString(PrevCanceled.Size());
            result += ", old failed: ";
            result += ToString(OldFailed.Size());
            result += ", prev failed: ";
            result += ToString(PrevFailed.Size());
            return result;
        }

    private:
        template <class T>
        static void CheckedInsert(T* where, const ui64 what) {
            Y_ASSERT(!where->Has(what));
            where->Insert(what);
            Y_ASSERT(where->Has(what));
        }
        template <class T>
        static void CheckedErase(T* from, const ui64 what) {
            Y_ASSERT(from->Has(what));
            from->Erase(what);
            Y_ASSERT(!from->Has(what));
        }

        bool IsInCurrent(const ui64 id) const {
            Y_ASSERT(id <= MaxSeenId);
            return MaxSeenId > 0 && id > MaxSeenId - Current.Size();
        }

        ui64 CurrentFrontId() const {
            Y_ASSERT(MaxSeenId > 0);
            return MaxSeenId - Current.Size() + 1;
        }

        char& CurrentState(const ui64 id) {
            Y_ASSERT(IsInCurrent(id));
            return Current[Current.Size() - 1 - (MaxSeenId - id)];
        }
        const char& CurrentState(const ui64 id) const {
            Y_ASSERT(IsInCurrent(id));
            return Current[Current.Size() - 1 - (MaxSeenId - id)];
        }

        void PushBackToCurrent(const char value) {
            if (Current.Full()) {
                const ui64 frontId = CurrentFrontId();
                switch (Current.Front()) {
                    case ES_HOLE:
                        Y_ASSERT(!PrevHoles.Has(frontId));
                        CheckedInsert(&OldHoles, frontId);
                        break;
                    case ES_ACTIVE:
                        CheckedInsert(&OldActive, frontId);
                        NumCurrentActive--;
                        break;
                    case ES_OK:
                        break;
                    case ES_FAILED:
                        Y_ASSERT(!PrevFailed.Has(frontId));
                        CheckedInsert(&OldFailed, frontId);
                        break;
                    case ES_CANCELED:
                        Y_ASSERT(!PrevCanceled.Has(frontId));
                        CheckedInsert(&OldCanceled, frontId);
                        break;
                    default:
                        Y_ASSERT(false);
                }
                Current.PopFront();
            }

            if (value == ES_ACTIVE) {
                NumCurrentActive++;
            }

            const char* r = Current.PushBack(value);
            Y_ASSERT(r);
            Y_UNUSED(r);

            MaxSeenId++;
            Y_ASSERT(CurrentState(MaxSeenId) == value);
        }

        ui64 MinSeenId;
        ui64 MaxSeenId;
        TCircularPodBuffer<char> Current; // contains info about (MaxSeenId - Current.Size(), MaxSeenId]
        size_t NumCurrentActive;
        TDisjointIntervalTree<ui64> OldActive; // contains active ids in [MinSeenId to MaxSeenId - Current.Size())
        TDisjointIntervalTree<ui64> OldHoles;  // contains hole ids in [MinSeenId to MaxSeenId - Current.Size())
        TDisjointIntervalTree<ui64> PrevHoles; // contains hole ids in [MinSeenId to MaxSeenId - Current.Size()) from previous step
        THashSetWithMin<ui64> OldCanceled;
        THashSetWithMin<ui64> PrevCanceled;
        THashSetWithMin<ui64> OldFailed;
        THashSetWithMin<ui64> PrevFailed;
        TInstant PrevHolesCreationTime;
    };
}
