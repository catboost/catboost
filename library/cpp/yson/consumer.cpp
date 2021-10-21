#include "consumer.h"
#include "string.h"
#include "parser.h"

namespace NYson {
    ////////////////////////////////////////////////////////////////////////////////


    void IYsonConsumer::OnRaw(const TYsonStringBuf& yson) {
        OnRaw(yson.AsStringBuf(), yson.GetType());
    }

    void TYsonConsumerBase::OnRaw(const TStringBuf& yson, EYsonType type) {
        ParseYsonStringBuffer(yson, this, type);
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace NYson
