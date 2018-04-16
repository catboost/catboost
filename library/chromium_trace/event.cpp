#include "event.h"

#include "gettime.h"

#include <util/system/getpid.h>
#include <util/system/thread.i>

namespace NChromiumTrace {
    TEventOrigin TEventOrigin::Here() {
        return TEventOrigin{
            GetPID(),
            SystemCurrentThreadIdImpl(),
        };
    }

    bool operator==(const TEventOrigin& lhs, const TEventOrigin& rhs) {
        return lhs.ProcessId == rhs.ProcessId && lhs.ThreadId == rhs.ThreadId;
    }

    TEventTime TEventTime::Now() {
        return TEventTime{
            GetWallTime(),
            GetThreadCPUTime(),
        };
    }

    bool operator==(const TEventTime& lhs, const TEventTime& rhs) {
        return lhs.WallTime == rhs.WallTime && lhs.ThreadCPUTime == rhs.ThreadCPUTime;
    }

    bool operator==(const TEventFlow& lhs, const TEventFlow& rhs) {
        return lhs.Type == rhs.Type && lhs.BindId == rhs.BindId;
    }

    bool operator==(const TEventArgs::TArg& lhs, const TEventArgs::TArg& rhs) {
        return lhs.Name == rhs.Name && lhs.Value == rhs.Value;
    }

    bool operator==(const TEventArgs& lhs, const TEventArgs& rhs) {
        return lhs.Items == rhs.Items;
    }

    bool operator==(const TDurationBeginEvent& lhs, const TDurationBeginEvent& rhs) {
        return lhs.Origin == rhs.Origin && lhs.Name == rhs.Name && lhs.Categories == rhs.Categories && lhs.Time == rhs.Time && lhs.Flow == rhs.Flow;
    }

    bool operator==(const TDurationEndEvent& lhs, const TDurationEndEvent& rhs) {
        return lhs.Origin == rhs.Origin && lhs.Time == rhs.Time && lhs.Flow == rhs.Flow;
    }

    bool operator==(const TDurationCompleteEvent& lhs, const TDurationCompleteEvent& rhs) {
        return lhs.Origin == rhs.Origin && lhs.Name == rhs.Name && lhs.Categories == rhs.Categories && lhs.BeginTime == rhs.BeginTime && lhs.EndTime == rhs.EndTime && lhs.Flow == rhs.Flow;
    }

    bool operator==(const TInstantEvent& lhs, const TInstantEvent& rhs) {
        return lhs.Origin == rhs.Origin && lhs.Name == rhs.Name && lhs.Categories == rhs.Categories && lhs.Time == rhs.Time && lhs.Scope == rhs.Scope;
    }

    bool operator==(const TAsyncEvent& lhs, const TAsyncEvent& rhs) {
        return lhs.SubType == rhs.SubType && lhs.Origin == rhs.Origin && lhs.Name == rhs.Name && lhs.Categories == rhs.Categories && lhs.Time == rhs.Time && lhs.Id == rhs.Id;
    }

    bool operator==(const TCounterEvent& lhs, const TCounterEvent& rhs) {
        return lhs.Origin == rhs.Origin && lhs.Name == rhs.Name && lhs.Categories == rhs.Categories && lhs.Time == rhs.Time;
    }

    bool operator==(const TMetadataEvent& lhs, const TMetadataEvent& rhs) {
        return lhs.Origin == rhs.Origin && lhs.Name == rhs.Name;
    }

    bool operator==(const TEventWithArgs& lhs, const TEventWithArgs& rhs) {
        return lhs.Event == rhs.Event && lhs.Args == rhs.Args;
    }
}
