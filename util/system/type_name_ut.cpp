#include "type_name.h"

#include <library/cpp/testing/unittest/registar.h>

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
        UNIT_ASSERT_VALUES_EQUAL(TypeName(value), "void");
#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&value), "void * __ptr64");
#else
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&value), "void*");
#endif

        int zero = 0;
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&zero), "int");

        double pi = M_PI;
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&pi), "double");

        std::string string;
        std::runtime_error err("This is awful");
#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&string), "class std::__y1::basic_string<char,struct std::__y1::char_traits<char>,class std::__y1::allocator<char> >");
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&err), "class std::runtime_error");
#else
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&string), "std::__y1::basic_string<char, std::__y1::char_traits<char>, std::__y1::allocator<char> >");
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&err), "std::runtime_error");
#endif
    }

    Y_UNIT_TEST(FromUserValues) {
        TSonde sonde;
        TRombicTail rombicTail;
        TRombicHead& castedTail = rombicTail;
#ifdef _MSC_VER
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&sonde), "class NUtil::NTypeNameTest::TSonde");
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&rombicTail), "class NUtil::NTypeNameTest::TRombicTail");
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&castedTail), "class NUtil::NTypeNameTest::TRombicTail");
#else
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&sonde), "NUtil::NTypeNameTest::TSonde");
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&rombicTail), "NUtil::NTypeNameTest::TRombicTail");
        UNIT_ASSERT_VALUES_EQUAL(TypeName(&castedTail), "NUtil::NTypeNameTest::TRombicTail");
#endif
    }

    Y_UNIT_TEST(FromTypeInfo) {
        UNIT_ASSERT_VALUES_EQUAL(TypeName(typeid(int)), "int");
        UNIT_ASSERT_VALUES_EQUAL(TypeName(std::type_index(typeid(int))), "int");
    }
}
