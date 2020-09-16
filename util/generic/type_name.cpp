#include "type_name.h"

#include <util/system/demangle.h>

TString TypeName(const std::type_info& typeInfo) {
    return CppDemangle(typeInfo.name());
}

TString TypeName(const std::type_index& typeIndex) {
    return CppDemangle(typeIndex.name());
}
