#include "maybe.h"

[[noreturn]] void NMaybe::TPolicyUndefinedExcept::OnEmpty() {
    ythrow yexception() << AsStringBuf("TMaybe is empty");
}

[[noreturn]] void NMaybe::TPolicyUndefinedFail::OnEmpty() {
    Y_FAIL("TMaybe is empty");
}

template <>
void Out<TNothing>(IOutputStream& o, const TNothing&) {
    o << "(empty maybe)";
}
