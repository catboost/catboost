#include "strbuf.h"

#include <ostream>

std::ostream& operator<<(std::ostream& os, TStringBuf buf) {
    os.write(buf.data(), buf.size());
    return os;
}
