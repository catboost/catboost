#ifndef ERROR_ATTRIBUTE_INL_H_
#error "Direct inclusion of this file is not allowed, include error_attribute.h"
// For the sake of sane code completion.
#include "error_attribute.h"
#endif

#include "text_yson.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NAttributeValueConversionImpl {

template <CPrimitiveConvertible T>
std::string TagInvoke(TTagInvokeTag<ToErrorAttributeValue>, const T& value)
{
    if constexpr (std::constructible_from<TStringBuf, const T&>) {
        return NDetail::ConvertToTextYsonString(TStringBuf(value));
    } else {
        return NDetail::ConvertToTextYsonString(value);
    }
}

////////////////////////////////////////////////////////////////////////////////

template <CPrimitiveConvertible T>
T TagInvoke(TFrom<T>, TStringBuf value)
{
    YT_VERIFY(!NDetail::IsBinaryYson(value));
    return NDetail::ConvertFromTextYsonString<T>(value);
}

} // namespace NAttributeValueConversionImpl

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
