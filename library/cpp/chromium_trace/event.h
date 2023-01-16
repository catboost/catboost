#pragma once

#include <library/cpp/containers/stack_vector/stack_vec.h>

#include <util/datetime/base.h>
#include <util/generic/typetraits.h>
#include <util/generic/variant.h>
#include <util/system/getpid.h>

namespace NChromiumTrace {
    struct TEventOrigin {
        TProcessId ProcessId;
        size_t ThreadId;

        static TEventOrigin Here();
    };

    bool operator==(const TEventOrigin& lhs, const TEventOrigin& rhs);
    static inline bool operator!=(const TEventOrigin& lhs, const TEventOrigin& rhs) {
        return !(lhs == rhs);
    }

    struct TEventTime {
        TInstant WallTime;
        TInstant ThreadCPUTime;

        static TEventTime Now();
    };

    bool operator==(const TEventTime& lhs, const TEventTime& rhs);
    static inline bool operator!=(const TEventTime& lhs, const TEventTime& rhs) {
        return !(lhs == rhs);
    }

    // With Flow API v2, first-class Flow events are deprecated
    // and replaced with flow information in regular events
    // See:
    // - Design doc: https://docs.google.com/document/d/1La_0PPfsTqHJihazYhff96thhjPtvq1KjAUOJu0dvEg
    // - Support in TraceViewer: https://github.com/catapult-project/catapult/commit/4f2e77ad98a5ea364d933515648260079bbc2bd2
    // - Support in Chromium: https://codereview.chromium.org/1239593002/diff/320001/base/trace_event/trace_event_impl.cc
    enum class EFlowType : ui8 {
        None = 0,
        Producer = 1,
        Consumer = 2,
        Step = Producer | Consumer,
    };

    inline EFlowType operator|(EFlowType lhs, EFlowType rhs) {
        return static_cast<EFlowType>(static_cast<ui8>(lhs) | static_cast<ui8>(rhs));
    }

    struct TEventFlow {
        EFlowType Type;
        ui64 BindId;
    };

    bool operator==(const TEventFlow& lhs, const TEventFlow& rhs);
    static inline bool operator!=(const TEventFlow& lhs, const TEventFlow& rhs) {
        return !(lhs == rhs);
    }

    struct TEventArgs {
        struct TArg {
            using TValue = std::variant<TStringBuf, i64, double>;

            TStringBuf Name;
            TValue Value;

            TArg()
                : Name()
                , Value(i64())
            {
            }

            template <typename T>
            TArg(TStringBuf name, T value)
                : Name(name)
                , Value(value)
            {
            }
        };

        TStackVec<TArg, 2> Items;

        TEventArgs() {
        }

        template <typename T>
        TEventArgs& Add(TStringBuf name, T value) {
            Items.emplace_back(name, value);
            return *this;
        }
    };

    bool operator==(const TEventArgs::TArg& lhs, const TEventArgs::TArg& rhs);
    static inline bool operator!=(const TEventArgs::TArg& lhs, const TEventArgs::TArg& rhs) {
        return !(lhs == rhs);
    }

    bool operator==(const TEventArgs& lhs, const TEventArgs& rhs);
    static inline bool operator!=(const TEventArgs& lhs, const TEventArgs& rhs) {
        return !(lhs == rhs);
    }

    struct TDurationBeginEvent {
        TEventOrigin Origin;
        TStringBuf Name;
        TStringBuf Categories;
        TEventTime Time;
        TEventFlow Flow;
    };

    bool operator==(const TDurationBeginEvent& lhs, const TDurationBeginEvent& rhs);
    static inline bool operator!=(const TDurationBeginEvent& lhs, const TDurationBeginEvent& rhs) {
        return !(lhs == rhs);
    }

    struct TDurationEndEvent {
        TEventOrigin Origin;
        TEventTime Time;
        TEventFlow Flow;
    };

    bool operator==(const TDurationEndEvent& lhs, const TDurationEndEvent& rhs);
    static inline bool operator!=(const TDurationEndEvent& lhs, const TDurationEndEvent& rhs) {
        return !(lhs == rhs);
    }

    struct TDurationCompleteEvent {
        TEventOrigin Origin;
        TStringBuf Name;
        TStringBuf Categories;
        TEventTime BeginTime;
        TEventTime EndTime;
        TEventFlow Flow;
    };

    bool operator==(const TDurationCompleteEvent& lhs, const TDurationCompleteEvent& rhs);
    static inline bool operator!=(const TDurationCompleteEvent& lhs, const TDurationCompleteEvent& rhs) {
        return !(lhs == rhs);
    }

    enum class EScope : ui8 {
        Thread,
        Process,
        Global,
    };

    struct TInstantEvent {
        TEventOrigin Origin;
        TStringBuf Name;
        TStringBuf Categories;
        TEventTime Time;
        EScope Scope;
    };

    bool operator==(const TInstantEvent& lhs, const TInstantEvent& rhs);
    static inline bool operator!=(const TInstantEvent& lhs, const TInstantEvent& rhs) {
        return !(lhs == rhs);
    }

    enum class EAsyncEvent : ui8 {
        Begin,
        End,
        Instant,
    };

    struct TAsyncEvent {
        EAsyncEvent SubType;
        TEventOrigin Origin;
        TStringBuf Name;
        TStringBuf Categories;
        TEventTime Time;
        ui64 Id;
    };

    bool operator==(const TAsyncEvent& lhs, const TAsyncEvent& rhs);
    static inline bool operator!=(const TAsyncEvent& lhs, const TAsyncEvent& rhs) {
        return !(lhs == rhs);
    }

    struct TCounterEvent {
        TEventOrigin Origin;
        TStringBuf Name;
        TStringBuf Categories;
        TEventTime Time;
    };

    bool operator==(const TCounterEvent& lhs, const TCounterEvent& rhs);
    static inline bool operator!=(const TCounterEvent& lhs, const TCounterEvent& rhs) {
        return !(lhs == rhs);
    }

    struct TMetadataEvent {
        TEventOrigin Origin;
        TStringBuf Name;
    };

    bool operator==(const TMetadataEvent& lhs, const TMetadataEvent& rhs);
    static inline bool operator!=(const TMetadataEvent& lhs, const TMetadataEvent& rhs) {
        return !(lhs == rhs);
    }

    using TAnyEvent = std::variant<
        TDurationBeginEvent,
        TDurationEndEvent,
        TDurationCompleteEvent,
        TInstantEvent,
        TAsyncEvent,
        TCounterEvent,
        TMetadataEvent>;

    struct TEventWithArgs {
        TAnyEvent Event;
        TEventArgs Args;

        TEventWithArgs()
            : Event(TMetadataEvent()) // XXX: Just to have some default constructor
        {
        }

        template <typename T>
        explicit TEventWithArgs(const T& event, const TEventArgs& args = TEventArgs())
            : Event(event)
            , Args(args)
        {
        }
    };

    bool operator==(const TEventWithArgs& lhs, const TEventWithArgs& rhs);
    static inline bool operator!=(const TEventWithArgs& lhs, const TEventWithArgs& rhs) {
        return !(lhs == rhs);
    }
}

Y_DECLARE_PODTYPE(NChromiumTrace::TEventOrigin);
Y_DECLARE_PODTYPE(NChromiumTrace::TEventTime);
Y_DECLARE_PODTYPE(NChromiumTrace::TEventFlow);
Y_DECLARE_PODTYPE(NChromiumTrace::TDurationEndEvent);
