#include "guid.h"

#include "format.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

void FormatValue(TStringBuilderBase* builder, TGuid value, TStringBuf /*spec*/)
{
    char* begin = builder->Preallocate(MaxGuidStringSize);
    char* end = WriteGuidToBuffer(begin, value);
    builder->Advance(end - begin);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

