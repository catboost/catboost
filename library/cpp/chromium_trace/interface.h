#pragma once

#include "global.h"
#include "guard.h"

#include <util/system/compat.h>

#define CHROMIUM_TRACE_GUARD_NAME current_yabs_trace_guard__

#define CHROMIUM_TRACE_COMPLETE_IMPL(name__, cat__)                  \
    ::NChromiumTrace::TCompleteEventGuard CHROMIUM_TRACE_GUARD_NAME( \
        ::NChromiumTrace::GetGlobalTracer(),                         \
        name__, cat__)

#define CHROMIUM_TRACE_COMPLETE_W_ARGS(name__, cat__, args)          \
    ::NChromiumTrace::TCompleteEventGuard CHROMIUM_TRACE_GUARD_NAME( \
        ::NChromiumTrace::GetGlobalTracer(),                         \
        name__, cat__, args)

#define CHROMIUM_TRACE_DURATION_IMPL(name__, cat__)                  \
    ::NChromiumTrace::TDurationEventGuard CHROMIUM_TRACE_GUARD_NAME( \
        ::NChromiumTrace::GetGlobalTracer(),                         \
        name__, cat__)

#define CHROMIUM_TRACE_DURATION_W_ARGS(name__, cat__, args)          \
    ::NChromiumTrace::TDurationEventGuard CHROMIUM_TRACE_GUARD_NAME( \
        ::NChromiumTrace::GetGlobalTracer(),                         \
        name__, cat__, args)

#define CHROMIUM_TRACE_SET_IN_FLOW(bind_id__) CHROMIUM_TRACE_GUARD_NAME.SetInFlow(bind_id__)
#define CHROMIUM_TRACE_SET_OUT_FLOW(bind_id__) CHROMIUM_TRACE_GUARD_NAME.SetOutFlow(bind_id__)

#define CHROMIUM_TRACE_FUNCTION_STR(name_str__) CHROMIUM_TRACE_COMPLETE_IMPL(name_str__, TStringBuf("func"))
#define CHROMIUM_TRACE_FUNCTION_NAME(name__) CHROMIUM_TRACE_FUNCTION_STR(TStringBuf(name__))
#define CHROMIUM_TRACE_FUNCTION() CHROMIUM_TRACE_FUNCTION_NAME(Y_FUNC_SIGNATURE)

#define CHROMIUM_TRACE_SCOPE_STR(name_str__) CHROMIUM_TRACE_COMPLETE_IMPL(name_str__, TStringBuf("scope"))
#define CHROMIUM_TRACE_SCOPE(name__) CHROMIUM_TRACE_SCOPE_STR(TStringBuf(name__))

#define CHROMIUM_TRACE_THREAD_NAME_STR(name__) ::NChromiumTrace::GetGlobalTracer()->AddCurrentThreadName(name__)
#define CHROMIUM_TRACE_THREAD_NAME(name__) CHROMIUM_TRACE_THREAD_NAME_STR(TStringBuf(name__))

#define CHROMIUM_TRACE_THREAD_INDEX(index__) ::NChromiumTrace::GetGlobalTracer()->AddCurrentThreadIndex(index__)
