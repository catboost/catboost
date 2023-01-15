#pragma once

#include "string.h"

#include <util/system/demangle.h>

#include <typeinfo>

//return human readable type name

//static type
template <class T>
static inline TString TypeName() {
    return CppDemangle(typeid(T).name());
}

//dynamic type
template <class T>
static inline TString TypeName(T* t) {
    (void)t;
    return CppDemangle(typeid(*t).name());
}

static inline TString TypeName(void*) {
    return "void";
}

static inline TString TypeName(const void*) {
    return "const void";
}
