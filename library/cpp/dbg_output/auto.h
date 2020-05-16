#pragma once

#include <util/generic/va_args.h>

// int a = 1, b = 2; Cout << LabeledDump(a, b, 1 + 2); yields {"a": 1, "b": 2, "1 + 2": 3}
#define LabeledDump(...) \
    '{' Y_PASS_VA_ARGS(Y_MAP_ARGS_WITH_LAST(__LABELED_DUMP_NONLAST__, __LABELED_DUMP_IMPL__, __VA_ARGS__)) << '}'
#define __LABELED_DUMP_IMPL__(x) << "\"" #x "\": " << DbgDump(x)
#define __LABELED_DUMP_NONLAST__(x) __LABELED_DUMP_IMPL__(x) << ", "

// Usage: struct TMyStruct { int A, B; }; DEFINE_DUMPER(TMyStruct, A, B); Cout << TMyStruct{3, 4};
// yields {"A": 3, "B": 4}
#define DEFINE_DUMPER(C, ...)                                                                                                                       \
    template <>                                                                                                                                     \
    struct TDumper<C> {                                                                                                                             \
        template <class S>                                                                                                                          \
        static inline void Dump(S& s, const C& v) {                                                                                                 \
            s << DumpRaw("{") Y_PASS_VA_ARGS(Y_MAP_ARGS_WITH_LAST(__DEFINE_DUMPER_NONLAST__, __DEFINE_DUMPER_IMPL__, __VA_ARGS__)) << DumpRaw("}"); \
        }                                                                                                                                           \
    };
#define __DEFINE_DUMPER_IMPL__(x) << DumpRaw("\"" #x "\": ") << v.x
#define __DEFINE_DUMPER_NONLAST__(x) __DEFINE_DUMPER_IMPL__(x) << DumpRaw(", ")
