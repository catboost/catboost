#include "maybe.h"

template <>
void Out<TNothing>(IOutputStream& o, const TNothing&) {
    o << "(empty maybe)";
}
