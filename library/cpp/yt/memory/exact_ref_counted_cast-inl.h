#ifndef EXACT_REF_COUNTED_CAST_INL_H_
#error "Direct inclusion of this file is not allowed, include exact_ref_counted_cast.h"
// For the sake of sane code completion.
#include "exact_ref_counted_cast.h"
#endif

#include "new.h"

#include <typeinfo>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T, class U>
std::conditional_t<std::is_const_v<U>, const T, T>* ExactRefCountedCast(U* ptr) noexcept
{
    static_assert(std::is_polymorphic_v<U>,
        "ExactRefCountedCast requires a polymorphic operand type");
    static_assert(std::is_base_of_v<TRefCountedBase, T>,
        "ExactRefCountedCast target must derive from TRefCountedBase");
    static_assert(std::is_final_v<TRefCountedWrapper<T>>,
        "TRefCountedWrapper must stay final for this cast to remain O(1)");

    using TWrapper = std::conditional_t<std::is_const_v<U>, const TRefCountedWrapper<T>, TRefCountedWrapper<T>>;
    // Wrapper is final, so dynamic_cast lowers to one type_info compare.
    return dynamic_cast<TWrapper*>(ptr);
}

template <class T, class U>
std::conditional_t<std::is_const_v<U>, const T, T>& ExactRefCountedCast(U& ref)
{
    static_assert(std::is_polymorphic_v<U>,
        "ExactRefCountedCast requires a polymorphic operand type");
    static_assert(std::is_base_of_v<TRefCountedBase, T>,
        "ExactRefCountedCast target must derive from TRefCountedBase");
    static_assert(std::is_final_v<TRefCountedWrapper<T>>,
        "TRefCountedWrapper must stay final for this cast to remain O(1)");

    using TWrapper = std::conditional_t<std::is_const_v<U>, const TRefCountedWrapper<T>, TRefCountedWrapper<T>>;
    return dynamic_cast<TWrapper&>(ref);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
