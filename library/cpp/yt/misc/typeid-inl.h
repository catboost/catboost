#ifndef TYPEID_INL_H_
#error "Direct inclusion of this file is not allowed, include typeid.h"
// For the sake of sane code completion.
#include "typeid.h"
#endif

#include "port.h"

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T>
class TTypeidTag
{ };

} // namespace NDetail

template <class T>
const std::type_info& Typeid()
{
    if constexpr (requires { TypeidImpl(NDetail::TTypeidTag<T>()); }) {
        return TypeidImpl(NDetail::TTypeidTag<T>());
    } else {
        return typeid(T);
    }
}

////////////////////////////////////////////////////////////////////////////////

#undef YT_DECLARE_TYPEID
#undef YT_DEFINE_TYPEID

#define YT_DECLARE_TYPEID(type) \
    [[maybe_unused]] YT_ATTRIBUTE_USED const std::type_info& TypeidImpl(::NYT::NDetail::TTypeidTag<type>);

#define YT_DEFINE_TYPEID(type) \
    [[maybe_unused]] YT_ATTRIBUTE_USED Y_FORCE_INLINE const std::type_info& TypeidImpl(::NYT::NDetail::TTypeidTag<type>) \
    { \
        return typeid(type); \
    }

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
