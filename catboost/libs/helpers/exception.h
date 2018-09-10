#pragma once

#include <util/generic/yexception.h>

class TCatboostException : public yexception {
};

#define CB_ENSURE_IMPL_1(CONDITION) Y_ENSURE_EX(CONDITION, TCatboostException() << AsStringBuf("Condition violated: `" Y_STRINGIZE(CONDITION) "'"))
#define CB_ENSURE_IMPL_2(CONDITION, MESSAGE) Y_ENSURE_EX(CONDITION, TCatboostException() << MESSAGE)

#define CB_ENSURE(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, CB_ENSURE_IMPL_2, CB_ENSURE_IMPL_1)(__VA_ARGS__))
