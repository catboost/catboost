#pragma once

#include <type_traits>

template <class JNITYPE, class CPPTYPE>
inline JNITYPE* ToJniPtr(CPPTYPE* cppPtr) {
    using NONCONST_CPPTYPE = std::remove_const_t<CPPTYPE>;
    if constexpr (std::is_same_v<JNITYPE, NONCONST_CPPTYPE>) {
        return const_cast<NONCONST_CPPTYPE*>(cppPtr);
    } else if constexpr (std::is_integral_v<JNITYPE> &&
        std::is_integral_v<NONCONST_CPPTYPE> &&
        (std::is_signed_v<JNITYPE> == std::is_signed_v<NONCONST_CPPTYPE>))
    {
        static_assert(sizeof(JNITYPE) == sizeof(NONCONST_CPPTYPE));
        return reinterpret_cast<JNITYPE*>(const_cast<NONCONST_CPPTYPE*>(cppPtr));
    }
}
