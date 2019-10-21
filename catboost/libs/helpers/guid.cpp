#include "guid.h"

#include <cstring>

NCB::TGuid NCB::CreateGuid() {
    TGUID guid;
    CreateGuid(&guid);
    return TGuid(guid);
}

bool NCB::TGuid::operator==(const NCB::TGuid& rhs) const {
    return Value == rhs.Value;
}

bool NCB::TGuid::operator!=(const NCB::TGuid& rhs) const {
    return !(*this == rhs);
}

bool NCB::TGuid::operator<(const NCB::TGuid& rhs) const {
    return Value < rhs.Value;
}

bool NCB::TGuid::operator>=(const NCB::TGuid& rhs) const {
    return !(*this < rhs);
}

bool NCB::TGuid::operator>(const NCB::TGuid& rhs) const {
    return (rhs < *this);
}

bool NCB::TGuid::operator<=(const NCB::TGuid& rhs) const {
    return !(*this > rhs);
}
