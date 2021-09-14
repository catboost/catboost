#pragma once

#include <util/generic/string.h>
#include <util/string/subst.h>

#include <typeindex>
#include <typeinfo>

// Consider using TypeName function family.
TString CppDemangle(const TString& name);

// TypeName function family return human readable type name.

TString TypeName(const std::type_info& typeInfo);
TString TypeName(const std::type_index& typeInfo);

// Works for types known at compile-time
// (thus, does not take any inheritance into account)
template <class T>
inline TString TypeName() {
    return TypeName(typeid(T));
}

// Works for dynamic type, including complex class hierarchies.
// Also, distinguishes between T, T*, T const*, T volatile*, T const volatile*,
// but not T and T const.
template <class T>
inline TString TypeName(const T& t) {
    return TypeName(typeid(t));
}
