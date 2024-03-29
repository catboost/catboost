#pragma once

#include <google/protobuf/stubs/common.h>

namespace google {
namespace protobuf {
namespace io {

void PrintJSONString(IOutputStream& stream, const TProtoStringType& string);

template<class T>
struct TAsJSON {
public:
    TAsJSON(const T& t)
        : T_(t)
    {
    }

    const T& T_;
};

template<class T>
inline IOutputStream& operator <<(IOutputStream& stream, const TAsJSON<T>& protoAsJSON) {
    protoAsJSON.T_.PrintJSON(stream);
    return stream;
}

}
}
}
