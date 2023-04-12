#pragma once

#include <util/system/yassert.h>

#include <library/cpp/testing/common/probe.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace testing {
    using NTesting::TProbe;
    using NTesting::TProbeState;
    using NTesting::TCoercibleToProbe;

    // A helper functor which extracts from probe-like objectss their state.
    struct TProbableTraits {
        static const TProbeState& ExtractState(const TProbeState& probe) {
            return probe;
        }

        static const TProbeState& ExtractState(const TProbeState* probe) {
            return *probe;
        }

        static const TProbeState& ExtractState(const TProbe& probe) {
            return *probe.State;
        }

        static const TProbeState& ExtractState(const TCoercibleToProbe& probe) {
            return *probe.State;
        }
    };

    void PrintTo(const TProbeState& state, ::std::ostream* os);

    inline void PrintTo(const TProbe& probe, ::std::ostream* os) {
        PrintTo(TProbableTraits::ExtractState(probe), os);
    }

    inline void PrintTo(const TCoercibleToProbe& probe, ::std::ostream* os) {
        PrintTo(TProbableTraits::ExtractState(probe), os);
    }

    MATCHER(IsAlive, "is alive") {
        Y_UNUSED(result_listener);
        const auto& state = TProbableTraits::ExtractState(arg);
        return state.Destructors < state.Constructors + state.CopyConstructors + state.CopyAssignments;
    }

    MATCHER(IsDead, "is dead") {
        Y_UNUSED(result_listener);
        const auto& state = TProbableTraits::ExtractState(arg);
        return state.Destructors == state.Constructors + state.CopyConstructors + state.CopyAssignments;
    }

    MATCHER_P2(HasCopyMoveCounts, copyCount, moveCount, "" + \
        PrintToString(copyCount) + " copy constructors and " + \
        PrintToString(moveCount) + " move constructors were called") {
        Y_UNUSED(result_listener);
        const auto& state = TProbableTraits::ExtractState(arg);
        return state.CopyConstructors == copyCount && state.MoveConstructors == moveCount;
    }

    MATCHER(NoCopies, "no copies were made") {
        Y_UNUSED(result_listener);
        const auto& state = TProbableTraits::ExtractState(arg);
        return 0 == state.CopyConstructors && 0 == state.CopyAssignments;
    }

    MATCHER(NoMoves, "no moves were made") {
        Y_UNUSED(result_listener);
        const auto& state = TProbableTraits::ExtractState(arg);
        return 0 == state.MoveConstructors && 0 == state.MoveAssignments;
    }

    MATCHER(NoAssignments, "no assignments were made") {
        Y_UNUSED(result_listener);
        const auto& state = TProbableTraits::ExtractState(arg);
        return 0 == state.CopyAssignments && 0 == state.MoveAssignments;
    }
}
