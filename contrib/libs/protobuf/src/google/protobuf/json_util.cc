#include <google/protobuf/json_util.h>

namespace google {
namespace protobuf {
namespace io {

void PrintJSONString(IOutputStream& stream, const TProtoStringType& string) {
    stream << '"';
    for (const char c: string) {
        switch(c) {
            case '"' : stream << "\\\""; continue;
            case '\\': stream << "\\\\"; continue;
            case '\b': stream << "\\b"; continue;
            case '\f': stream << "\\f"; continue;
            case '\n': stream << "\\n"; continue;
            case '\r': stream << "\\r"; continue;
            case '\t': stream << "\\t"; continue;
        }
        if ((unsigned char)c < 0x20) {
            stream << "\\u00";
            static const char hexDigits[] = "0123456789ABCDEF";
            stream << hexDigits[(c & 0xf0) >> 4];
            stream << hexDigits[(c & 0x0f)];
            continue;
        }
        stream << c;
    }
    stream << '"';
}

}
}
}
