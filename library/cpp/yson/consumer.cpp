#include "consumer.h"
#include "string.h"
#include "parser.h"

namespace NYson {

    ////////////////////////////////////////////////////////////////////////////////

    void TYsonConsumerBase::OnRaw(TStringBuf str, NYT::NYson::EYsonType type) {
        ParseYsonStringBuffer(str, this, type);
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
