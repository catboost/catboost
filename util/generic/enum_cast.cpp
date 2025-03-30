#include "enum_cast.h"

#include <util/generic/yexception.h>
#include <util/system/type_name.h>

namespace NPrivate {

    [[noreturn]] void OnSafeCastToEnumUnexpectedValue(const std::type_info& valueTypeInfo) {
        ythrow TBadCastException() << "Unexpected enum " << TypeName(valueTypeInfo) << " value";
    }

} // namespace NPrivate
