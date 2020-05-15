#pragma once
// Sample file for parse_enum unittests

#include <util/generic/fwd.h>
#include <util/system/compiler.h>

// Test template declarations
template<class T>
void Func(T&);

template<>
void Func(struct ENonDeclared&);

template<class TClass> class TFwdDecl;

// Test in-function class declarations
void InexistentFunction(struct TFwdStructDecl);
void InexistentFunction2(struct TFwdStructDecl, class TMegaClass);


static inline void Func() {
    class TLocal {
        int M;
    public:
        void F() {
            // to shut up clang
            Y_UNUSED(M);
        }
    };

    {
        // unnamed block
    }
}

// Test forward declarations, pt 2
namespace NTestContainer {
    struct TStruct;
}

// Enums
enum ESimple {
    Http,
    Https,
    ItemCount
};

enum class ESimpleWithComma {
    Http = 3,
    Http2 = Http,
    Https, // 4
    ItemCount, // 5
};

enum ECustomAliases {
    CAHttp = 3 /* "http" */,
    CAHttps /* "https" */,
    CAItemCount,
};

enum EMultipleAliases {
    MAHttp = 9 /* "http://" "secondary" "old\nvalue" */,
    MAHttps = 1 /* "https://" */,
    MAItemCount,
};

namespace NEnumNamespace {
    enum EInNamespace {
        Http = 9 /* "http://" "secondary" "old\nvalue" */,
        Https = 1 /* "https://" */,
        ItemCount /* "real value" */,
    };
};

struct TStruct {
    int M;
};

namespace NEnumNamespace {
    class TEnumClass: public TStruct {
    public:
        enum EInClass {
            Http = 9 /* "http://" "secondary" "old\nvalue" */,
            Https1 = NEnumNamespace::Https /* "https://" */,
            // holy crap, this will work too:
            Https3 = 1 /* "https://" */ + 2,
        };
    };
}

enum {
    One,
    Two,
    Three,
};

struct {
    int M;
} SomeStruct;

static inline void f() {
    (void)(SomeStruct);
    (void)(f);
}

// buggy case taken from library/cpp/html/face/parstypes.h
enum TEXT_WEIGHT {
    WEIGHT_ZERO=-1,// NOINDEX_RELEV
    WEIGHT_LOW,    // LOW_RELEV
    WEIGHT_NORMAL, // NORMAL_RELEV
    WEIGHT_HIGH,   // HIGH_RELEV   (H1,H2,H3,ADDRESS,CAPTION)
    WEIGHT_BEST    // BEST_RELEV   (TITLE)
};

// enum with duplicate keys
enum EDuplicateKeys {
    Key0 = 0,
    Key0Second = Key0,
    Key1,
    Key2 = 3 /* "k2" "k2.1" */,
    Key3 = 3 /* "k3" */,
};

enum class EFwdEnum;
void FunctionUsingEFwdEnum(EFwdEnum);
enum class EFwdEnum {
    One,
    Two
};

// empty enum (bug found by sankear@)
enum EEmpty {
};

namespace NComposite::NInner {
    enum EInCompositeNamespaceSimple {
        one,
        two = 2,
        three,
    };
}

namespace NOuterSimple {
    namespace NComposite::NMiddle::NInner {
        namespace NInnerSimple {
            class TEnumClass {
            public:
                enum EVeryDeep {
                    Key0 = 0,
                    Key1 = 1,
                };
            };
        }
    }
}


constexpr int func(int value) {
    return value;
}

#define MACRO(x, y) x

// enum with nonliteral values
enum ENonLiteralValues {
    one = MACRO(1, 2),
    two = 2,
    three = func(3),
    four,
    five = MACRO(MACRO(1, 2), 2),
};

#undef MACRO


enum EDestructionPriorityTest {
    first,
    second
};


enum class NotifyingStatus
{
    NEW = 0,
    FAILED_WILL_RETRY = 1,
    FAILED_NO_MORE_TRIALS = 2,
    SENT = 3
};

/*
 * Still unsupported features:
 *
 * a) Anonymous namespaces (it is parsed correctly, though)
 * b) Enums inside template classes (impossible by design)
 **/
