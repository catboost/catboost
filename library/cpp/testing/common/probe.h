#pragma once

#include <util/system/yassert.h>

namespace NTesting {
    ////////////////////////////////////////////////////////////////////////////////

    // Below there is a serie of probe classes for testing construction/destruction copying/moving of class.
    // for examples see tests in probe_ut.cpp

    struct TProbeState {
        int Constructors = 0;
        int Destructors = 0;
        int ShadowDestructors = 0;
        int CopyConstructors = 0;
        int CopyAssignments = 0;
        int MoveConstructors = 0;
        int MoveAssignments = 0;
        int Touches = 0;

        TProbeState() = default;

        void Reset() {
            *this = TProbeState{};
        }
    };

    // Used for probing the number of copies that occur if a type must be coerced.
    class TCoercibleToProbe {
    public:
        TProbeState* State;
        TProbeState* ShadowState;

    public:
        explicit TCoercibleToProbe(TProbeState* state)
            : State(state)
            , ShadowState(state)
        {}

    private:
        TCoercibleToProbe(const TCoercibleToProbe&);
        TCoercibleToProbe(TCoercibleToProbe&&);
        TCoercibleToProbe& operator=(const TCoercibleToProbe&);
        TCoercibleToProbe& operator=(TCoercibleToProbe&&);
    };

    // Used for probing the number of copies in an argument.
    class TProbe {
    public:
        TProbeState* State;
        TProbeState* ShadowState;

    public:
        static TProbe ExplicitlyCreateInvalidProbe() {
            return TProbe();
        }

        explicit TProbe(TProbeState* state)
            : State(state)
            , ShadowState(state)
        {
            Y_ASSERT(State);
            ++State->Constructors;
        }

        ~TProbe() {
            if (State) {
                ++State->Destructors;
            }
            if (ShadowState) {
                ++ShadowState->ShadowDestructors;
            }
        }

        TProbe(const TProbe& other)
            : State(other.State)
            , ShadowState(other.ShadowState)
        {
            Y_ASSERT(State);
            ++State->CopyConstructors;
        }

        TProbe(TProbe&& other)
            : State(other.State)
            , ShadowState(other.ShadowState)
        {
            Y_ASSERT(State);
            other.State = nullptr;
            ++State->MoveConstructors;
        }

        TProbe(const TCoercibleToProbe& other)
            : State(other.State)
            , ShadowState(other.ShadowState)
        {
            Y_ASSERT(State);
            ++State->CopyConstructors;
        }

        TProbe(TCoercibleToProbe&& other)
            : State(other.State)
            , ShadowState(other.ShadowState)
        {
            Y_ASSERT(State);
            other.State = nullptr;
            ++State->MoveConstructors;
        }

        TProbe& operator=(const TProbe& other) {
            State = other.State;
            ShadowState = other.ShadowState;
            Y_ASSERT(State);
            ++State->CopyAssignments;
            return *this;
        }

        TProbe& operator=(TProbe&& other) {
            State = other.State;
            ShadowState = other.ShadowState;
            Y_ASSERT(State);
            other.State = nullptr;
            ++State->MoveAssignments;
            return *this;
        }

        void Touch() const {
            Y_ASSERT(State);
            ++State->Touches;
        }

        bool IsValid() const {
            return nullptr != State;
        }

    private:
        TProbe()
            : State(nullptr)
        {}
    };
} // namespace NTesting
