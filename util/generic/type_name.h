#pragma once

#include "string.h"

#include <typeindex>
#include <typeinfo>

// TypeName function family return human readable type name.

TString TypeName(const std::type_info& typeInfo);
TString TypeName(const std::type_index& typeInfo);

// Works for types known at compile-time
// (thus, does not take any inheritance into account)
template <class T>
inline TString TypeName() {
    return TypeName(typeid(T));
}

// Works for dynamic type, including complex class hierarchies
// (note that values must be passed by pointer).
template <class T>
inline TString TypeName(T* t) {
    return TypeName(typeid(*t));
}

// ISO C++ does not allow indirection on operand of type 'void *'
inline TString TypeName(void*) {
    return "void";
}

inline TString TypeName(const void*) {
    return "const void";
}
