#pragma once

/// see enum_codegen_ut.cpp for examples

#define ENUM_VALUE_GEN(name, value, ...) name = value,
#define ENUM_VALUE_GEN_NO_VALUE(name, ...) name,

#define ENUM_TO_STRING_IMPL_ITEM(name, ...) \
    case name:                              \
        return #name;
#define ENUM_LTLT_IMPL_ITEM(name, ...) \
    case name:                         \
        os << #name;                   \
        break;

#define ENUM_TO_STRING(type, MAP)                                            \
    static inline const char* ToCString(type value) {                        \
        switch (value) {                                                     \
            MAP(ENUM_TO_STRING_IMPL_ITEM)                                    \
            default:                                                         \
                return "UNKNOWN";                                            \
        }                                                                    \
    }                                                                        \
                                                                             \
    static inline IOutputStream& operator<<(IOutputStream& os, type value) { \
        switch (value) {                                                     \
            MAP(ENUM_LTLT_IMPL_ITEM)                                         \
            default:                                                         \
                os << int(value);                                            \
                break;                                                       \
        }                                                                    \
        return os;                                                           \
    }
