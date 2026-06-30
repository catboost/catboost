#include "maybe.h"
#include <util/system/type_name.h>

[[noreturn]] void NMaybe::TPolicyUndefinedExcept::OnEmpty(const std::type_info& valueTypeInfo) {
    ythrow yexception() << "TMaybe is empty, value type: "sv << TypeName(valueTypeInfo);
}

[[noreturn]] void NMaybe::TPolicyUndefinedFail::OnEmpty(const std::type_info& valueTypeInfo) {
    const TString typeName = TypeName(valueTypeInfo);
    Y_ABORT("TMaybe is empty, value type: %s", typeName.c_str());
}

template <>
void Out<TNothing>(IOutputStream& o, const TNothing&) {
    o << "(empty maybe)";
}
