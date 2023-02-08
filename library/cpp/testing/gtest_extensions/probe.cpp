#include "probe.h"

#include <ostream>

namespace testing {
    void PrintTo(const TProbeState& state, ::std::ostream* os) {
        int copies = state.CopyConstructors + state.CopyAssignments;
        int moves = state.MoveConstructors + state.MoveAssignments;
        *os << state.Constructors << " ctors, " << state.Destructors << " dtors; "
            << "copies: " << copies << " = " << state.CopyConstructors << " + " << state.CopyAssignments << "; "
            << "moves: " << moves << " = " << state.MoveConstructors << " + " << state.MoveAssignments;
    }
} // namespace testing
