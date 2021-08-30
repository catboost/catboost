
#include <library/cpp/chromium_trace/interface.h>

#include <util/datetime/base.h>


int main(int /*argc*/, const char* /*argv*/[]) {
    NChromiumTrace::TGlobalJsonFileSink traceSink("trace.json");

    {
        NChromiumTrace::TEventArgs eventArgs;
        eventArgs.Add("arg1", i64(42));
        eventArgs.Add("arg2", i64(43));
        eventArgs.Add("arg3", i64(44));
        CHROMIUM_TRACE_DURATION_W_ARGS("name1", "cat1", &eventArgs);
        Sleep(TDuration::Seconds(1));
    }
    {
        NChromiumTrace::TEventArgs eventArgs;
        eventArgs.Add("arg11", TStringBuf("val1"));
        CHROMIUM_TRACE_DURATION_W_ARGS("name2", "cat1", &eventArgs);
        Sleep(TDuration::Seconds(2));
    }
    return 0;
}
