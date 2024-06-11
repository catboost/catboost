#include "variant.h"

#include <library/cpp/yt/string/string_builder.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

void FormatValue(TStringBuilderBase* builder, const std::monostate&, TStringBuf /*spec*/)
{
    builder->AppendString(TStringBuf("<monostate>"));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
