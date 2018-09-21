#pragma once

#include <util/generic/yexception.h>

class TCatboostException : public yexception {
};

#define CB_ENSURE_IMPL_1(CONDITION) Y_ENSURE_EX(CONDITION, TCatboostException() << AsStringBuf("Condition violated: `" Y_STRINGIZE(CONDITION) "'"))
#define CB_ENSURE_IMPL_2(CONDITION, MESSAGE) Y_ENSURE_EX(CONDITION, TCatboostException() << MESSAGE)

#define CB_ENSURE(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, CB_ENSURE_IMPL_2, CB_ENSURE_IMPL_1)(__VA_ARGS__))


/* Internal errors are violations of some internal invariants and logic,
 * not caused by invalid user input or hardware limitations (e.g. lack of free RAM or disk space)
 * or failures (e.g. network errors)
 */

namespace NCB {

    extern const TStringBuf INTERNAL_ERROR_MSG;

}

#define CB_ENSURE_INTERNAL(CONDITION, MESSAGE) Y_ENSURE_EX( \
    CONDITION, \
    TCatboostException() << NCB::INTERNAL_ERROR_MSG << MESSAGE \
)
