#include "consumer.h"
#include "string.h"
#include "parser.h"

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////


    void IYsonConsumer::OnRaw(const TYsonStringBuf& yson) {
        OnRaw(yson.AsStringBuf(), yson.GetType());
    }

    void TYsonConsumerBase::OnRaw(TStringBuf str, EYsonType type) {
        ParseYsonStringBuffer(str, this, type);
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
