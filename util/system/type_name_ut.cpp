#include "type_name.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/yexception.h>
#include <util/generic/fwd.h>

#include <stdexcept>
#include <string>

Y_UNIT_TEST_SUITE(TDemangleTest) {
    Y_UNIT_TEST(SimpleTest) {
        // just check it does not crash or leak
        CppDemangle("hello");
        CppDemangle("");
        CppDemangle("Sfsdf$dfsdfTTSFSDF23234::SDFS:FSDFSDF#$%");
    }
}

namespace NUtil::NTypeNameTest {

    class TSonde {
        // intentionally left empty
    };

    class TRombicHead {
    public:
        virtual ~TRombicHead() = default;
    };

    class TRombicLeftArc : public virtual TRombicHead {
    public:
        int x;
        virtual ~TRombicLeftArc() = default;
    };

    class TRombicRightArc : public virtual TRombicHead {
    public:
        int y;
        virtual ~TRombicRightArc() = default;
    };

    class TRombicTail : public virtual TRombicRightArc, TRombicLeftArc {
    public:
        virtual ~TRombicTail() = default;
    };

    class TFromThis {
    public:
        TString GetTypeName() const {
            return BetterTypeName(*this);
        }
    };
}

using namespace NUtil::NTypeNameTest;

Y_UNIT_TEST_SUITE(TypeName) {

    Y_UNIT_TEST(FromWellKnownTypes) {
        UNIT_ASSERT_VALUES_EQUAL(TypeName<void>(), "void");
#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(TypeName<void*>(), "void * __ptr64");
#else
        UNIT_ASSERT_VALUES_EQUAL(TypeName<void*>(), "void*");
#endif
        UNIT_ASSERT_VALUES_EQUAL(TypeName<int>(), "int");
        UNIT_ASSERT_VALUES_EQUAL(TypeName<double>(), "double");

#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(TypeName<std::string>(), "class std::__y1::basic_string<char,struct std::__y1::char_traits<char>,class std::__y1::allocator<char> >");
        UNIT_ASSERT_VALUES_EQUAL(TypeName<std::runtime_error>(), "class std::runtime_error");
#else
        UNIT_ASSERT_VALUES_EQUAL(TypeName<std::string>(), "std::__y1::basic_string<char, std::__y1::char_traits<char>, std::__y1::allocator<char> >");
        UNIT_ASSERT_VALUES_EQUAL(TypeName<std::runtime_error>(), "std::runtime_error");
#endif
    }

    Y_UNIT_TEST(FromArcadiaTypes) {
#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(TypeName<yexception>(), "class yexception");
        UNIT_ASSERT_VALUES_EQUAL(TypeName<TString>(), "class TBasicString<char,struct std::__y1::char_traits<char> >");
#else
        UNIT_ASSERT_VALUES_EQUAL(TypeName<yexception>(), "yexception");
        UNIT_ASSERT_VALUES_EQUAL(TypeName<TString>(), "TBasicString<char, std::__y1::char_traits<char> >");
#endif
    }

    Y_UNIT_TEST(FromUserTypes) {
#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(TypeName<TSonde>(), "class NUtil::NTypeNameTest::TSonde");
        UNIT_ASSERT_VALUES_EQUAL(TypeName<TRombicTail>(), "class NUtil::NTypeNameTest::TRombicTail");
#else
        UNIT_ASSERT_VALUES_EQUAL(TypeName<TSonde>(), "NUtil::NTypeNameTest::TSonde");
        UNIT_ASSERT_VALUES_EQUAL(TypeName<TRombicTail>(), "NUtil::NTypeNameTest::TRombicTail");
#endif
    }

    Y_UNIT_TEST(FromWellKnownValues) {
        void* value = (void*)"123";
        const void* constValue = (const void*)"456";

#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(value), "void * __ptr64");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(&value), "void * __ptr64 * __ptr64");

        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(constValue), "void const * __ptr64");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(&constValue), "void const * __ptr64 * __ptr64");
#else
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(value), "void*");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(&value), "void**");

        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(constValue), "void const*");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(&constValue), "void const**");
#endif

        int zero = 0;
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(zero), "int");

        double pi = M_PI;
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(pi), "double");

        std::string string;
        std::runtime_error err("This is awful");
#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(string), "class std::__y1::basic_string<char,struct std::__y1::char_traits<char>,class std::__y1::allocator<char> >");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(err), "class std::runtime_error");
#else
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(string), "std::__y1::basic_string<char, std::__y1::char_traits<char>, std::__y1::allocator<char> >");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(err), "std::runtime_error");
#endif
    }

    Y_UNIT_TEST(FromUserValues) {
        TSonde sonde;
        const TSonde constSonde;

        TRombicTail rombicTail;
        TRombicHead& castedTail = rombicTail;

        TFromThis obj;

#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(sonde), "class NUtil::NTypeNameTest::TSonde");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(constSonde), "class NUtil::NTypeNameTest::TSonde");

        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(rombicTail), "class NUtil::NTypeNameTest::TRombicTail");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(castedTail), "class NUtil::NTypeNameTest::TRombicTail");

        UNIT_ASSERT_VALUES_EQUAL(obj.GetTypeName(), "class NUtil::NTypeNameTest::TFromThis");
#else
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(sonde), "NUtil::NTypeNameTest::TSonde");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(constSonde), "NUtil::NTypeNameTest::TSonde");

        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(rombicTail), "NUtil::NTypeNameTest::TRombicTail");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(castedTail), "NUtil::NTypeNameTest::TRombicTail");

        UNIT_ASSERT_VALUES_EQUAL(obj.GetTypeName(), "NUtil::NTypeNameTest::TFromThis");
#endif
    }

    Y_UNIT_TEST(FromTypeInfo) {
        UNIT_ASSERT_VALUES_EQUAL(TypeName(typeid(int)), "int");
        UNIT_ASSERT_VALUES_EQUAL(TypeName(std::type_index(typeid(int))), "int");
    }

    Y_UNIT_TEST(DistinguishPointerQualifiers) {
        char *simplePtr = nullptr;
        const char *constPtr = nullptr;
        volatile char *volatilePtr = nullptr;
        const volatile char *cvPtr = nullptr;

#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(simplePtr), "char * __ptr64");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(constPtr), "char const * __ptr64");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(volatilePtr), "char volatile * __ptr64");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(cvPtr), "char const volatile * __ptr64");
#else
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(simplePtr), "char*");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(constPtr), "char const*");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(volatilePtr), "char volatile*");
        UNIT_ASSERT_VALUES_EQUAL(BetterTypeName(cvPtr), "char const volatile*");
#endif
    }
}
