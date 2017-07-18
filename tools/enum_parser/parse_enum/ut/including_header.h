#pragma once

#include <util/generic/serialized_enum.h>
#include <tools/enum_parser/parse_enum/ut/enums_with_header.h_serialized.h>

int TestEnumWithHeader() {
    return GetEnumItemsCount<EWithHeader>();
}

