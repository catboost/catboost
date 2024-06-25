#include "enum_range.h"

#include <util/stream/output.h>
#include <util/system/defaults.h>
#include <library/cpp/testing/unittest/registar.h>

class TEnumRangeTest: public TTestBase {
    UNIT_TEST_SUITE(TEnumRangeTest);
    UNIT_TEST(TestGlobalEnumRange)
    UNIT_TEST(TestNamedNamespaceEnumRange)
    UNIT_TEST(TestAnonNamespaceEnumRange)
    UNIT_TEST(TestMemberEnumRange)
    UNIT_TEST_SUITE_END();

protected:
    void TestGlobalEnumRange();
    void TestNamedNamespaceEnumRange();
    void TestAnonNamespaceEnumRange();
    void TestMemberEnumRange();
};

UNIT_TEST_SUITE_REGISTRATION(TEnumRangeTest);

#define Y_DEFINE_ENUM_SERIALIZATION(Enum, Prefix)        \
    Y_DECLARE_OUT_SPEC(inline, Enum, stream, value) {    \
        switch (value) {                                 \
            using enum Enum;                             \
            case Y_CAT(Prefix, 1):                       \
                stream << Y_STRINGIZE(Y_CAT(Prefix, 1)); \
            case Y_CAT(Prefix, 2):                       \
                stream << Y_STRINGIZE(Y_CAT(Prefix, 2)); \
            case Y_CAT(Prefix, 3):                       \
                stream << Y_STRINGIZE(Y_CAT(Prefix, 3)); \
        }                                                \
    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class EGlobal1 {
    G11,
    G12,
    G13
};
Y_DEFINE_ENUM_MINMAX(EGlobal1, G11, G13);

enum EGlobal2 {
    G21 = 5,
    G22,
    G23 = 9
};
Y_DEFINE_ENUM_MINMAX(EGlobal2, G21, G23);

enum class EGlobal3 {
    G31,
    G32,
    G33
};
Y_DEFINE_ENUM_MAX(EGlobal3, G33);

enum EGlobal4 {
    G41,
    G42,
    G43
};
Y_DEFINE_ENUM_MAX(EGlobal4, G43);

Y_DEFINE_ENUM_SERIALIZATION(EGlobal1, G1);
Y_DEFINE_ENUM_SERIALIZATION(EGlobal2, G2);
Y_DEFINE_ENUM_SERIALIZATION(EGlobal3, G3);
Y_DEFINE_ENUM_SERIALIZATION(EGlobal4, G4);

void TEnumRangeTest::TestGlobalEnumRange() {
    UNIT_ASSERT_VALUES_EQUAL(EGlobal1::G11, TEnumRange<EGlobal1>::Min);
    UNIT_ASSERT_VALUES_EQUAL(EGlobal1::G13, TEnumRange<EGlobal1>::Max);
    UNIT_ASSERT_VALUES_EQUAL(EGlobal2::G21, TEnumRange<EGlobal2>::Min);
    UNIT_ASSERT_VALUES_EQUAL(EGlobal2::G23, TEnumRange<EGlobal2>::Max);
    UNIT_ASSERT_VALUES_EQUAL(EGlobal3::G31, TEnumRange<EGlobal3>::Min);
    UNIT_ASSERT_VALUES_EQUAL(EGlobal3::G33, TEnumRange<EGlobal3>::Max);
    UNIT_ASSERT_VALUES_EQUAL(EGlobal4::G41, TEnumRange<EGlobal4>::Min);
    UNIT_ASSERT_VALUES_EQUAL(EGlobal4::G43, TEnumRange<EGlobal4>::Max);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<EGlobal1>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<EGlobal1>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(5, TEnumRange<EGlobal2>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(9, TEnumRange<EGlobal2>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<EGlobal3>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<EGlobal3>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<EGlobal4>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<EGlobal4>::UnderlyingMax);
}

namespace NNamespace {
    enum class ENamed1 {
        N11,
        N12,
        N13
    };
    Y_DEFINE_ENUM_MINMAX(ENamed1, N11, N13);

    enum ENamed2 {
        N21 = 5,
        N22,
        N23 = 9
    };
    Y_DEFINE_ENUM_MINMAX(ENamed2, N21, N23);

    enum class ENamed3 {
        N31,
        N32,
        N33
    };
    Y_DEFINE_ENUM_MAX(ENamed3, N33);

    enum ENamed4 {
        N41,
        N42,
        N43
    };
    Y_DEFINE_ENUM_MAX(ENamed4, N43);
}

Y_DEFINE_ENUM_SERIALIZATION(NNamespace::ENamed1, N1);
Y_DEFINE_ENUM_SERIALIZATION(NNamespace::ENamed2, N2);
Y_DEFINE_ENUM_SERIALIZATION(NNamespace::ENamed3, N3);
Y_DEFINE_ENUM_SERIALIZATION(NNamespace::ENamed4, N4);

void TEnumRangeTest::TestNamedNamespaceEnumRange() {
    UNIT_ASSERT_VALUES_EQUAL(NNamespace::ENamed1::N11, TEnumRange<NNamespace::ENamed1>::Min);
    UNIT_ASSERT_VALUES_EQUAL(NNamespace::ENamed1::N13, TEnumRange<NNamespace::ENamed1>::Max);
    UNIT_ASSERT_VALUES_EQUAL(NNamespace::ENamed2::N21, TEnumRange<NNamespace::ENamed2>::Min);
    UNIT_ASSERT_VALUES_EQUAL(NNamespace::ENamed2::N23, TEnumRange<NNamespace::ENamed2>::Max);
    UNIT_ASSERT_VALUES_EQUAL(NNamespace::ENamed3::N31, TEnumRange<NNamespace::ENamed3>::Min);
    UNIT_ASSERT_VALUES_EQUAL(NNamespace::ENamed3::N33, TEnumRange<NNamespace::ENamed3>::Max);
    UNIT_ASSERT_VALUES_EQUAL(NNamespace::ENamed4::N41, TEnumRange<NNamespace::ENamed4>::Min);
    UNIT_ASSERT_VALUES_EQUAL(NNamespace::ENamed4::N43, TEnumRange<NNamespace::ENamed4>::Max);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<NNamespace::ENamed1>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<NNamespace::ENamed1>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(5, TEnumRange<NNamespace::ENamed2>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(9, TEnumRange<NNamespace::ENamed2>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<NNamespace::ENamed3>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<NNamespace::ENamed3>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<NNamespace::ENamed4>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<NNamespace::ENamed4>::UnderlyingMax);
}

namespace {
    enum class EAnon1 {
        A11,
        A12,
        A13
    };
    Y_DEFINE_ENUM_MINMAX(EAnon1, A11, A13);

    enum EAnon2 {
        A21 = 5,
        A22,
        A23 = 9
    };
    Y_DEFINE_ENUM_MINMAX(EAnon2, A21, A23);

    enum class EAnon3 {
        A31,
        A32,
        A33
    };
    Y_DEFINE_ENUM_MAX(EAnon3, A33);

    enum EAnon4 {
        A41,
        A42,
        A43
    };
    Y_DEFINE_ENUM_MAX(EAnon4, A43);
}

void TEnumRangeTest::TestAnonNamespaceEnumRange() {
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<EAnon1>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<EAnon1>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(5, TEnumRange<EAnon2>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(9, TEnumRange<EAnon2>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<EAnon3>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<EAnon3>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<EAnon4>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<EAnon4>::UnderlyingMax);
}

struct TTestStruct {
    enum class EMember1 {
        M11,
        M12,
        M13
    };
    Y_DEFINE_ENUM_MINMAX_FRIEND(EMember1, M11, M13);

    enum EMember2 {
        M21 = 5,
        M22,
        M23 = 9
    };
    Y_DEFINE_ENUM_MINMAX_FRIEND(EMember2, M21, M23);

    enum class EMember3 {
        M31,
        M32,
        M33
    };
    Y_DEFINE_ENUM_MAX_FRIEND(EMember3, M33);

    enum EMember4 {
        M41,
        M42,
        M43
    };
    Y_DEFINE_ENUM_MAX_FRIEND(EMember4, M43);
};

Y_DEFINE_ENUM_SERIALIZATION(TTestStruct::EMember1, M1);
Y_DEFINE_ENUM_SERIALIZATION(TTestStruct::EMember2, M2);
Y_DEFINE_ENUM_SERIALIZATION(TTestStruct::EMember3, M3);
Y_DEFINE_ENUM_SERIALIZATION(TTestStruct::EMember4, M4);

void TEnumRangeTest::TestMemberEnumRange() {
    UNIT_ASSERT_VALUES_EQUAL(TTestStruct::EMember1::M11, TEnumRange<TTestStruct::EMember1>::Min);
    UNIT_ASSERT_VALUES_EQUAL(TTestStruct::EMember1::M13, TEnumRange<TTestStruct::EMember1>::Max);
    UNIT_ASSERT_VALUES_EQUAL(TTestStruct::EMember2::M21, TEnumRange<TTestStruct::EMember2>::Min);
    UNIT_ASSERT_VALUES_EQUAL(TTestStruct::EMember2::M23, TEnumRange<TTestStruct::EMember2>::Max);
    UNIT_ASSERT_VALUES_EQUAL(TTestStruct::EMember3::M31, TEnumRange<TTestStruct::EMember3>::Min);
    UNIT_ASSERT_VALUES_EQUAL(TTestStruct::EMember3::M33, TEnumRange<TTestStruct::EMember3>::Max);
    UNIT_ASSERT_VALUES_EQUAL(TTestStruct::EMember4::M41, TEnumRange<TTestStruct::EMember4>::Min);
    UNIT_ASSERT_VALUES_EQUAL(TTestStruct::EMember4::M43, TEnumRange<TTestStruct::EMember4>::Max);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<TTestStruct::EMember1>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<TTestStruct::EMember1>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(5, TEnumRange<TTestStruct::EMember2>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(9, TEnumRange<TTestStruct::EMember2>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<TTestStruct::EMember3>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<TTestStruct::EMember3>::UnderlyingMax);
    UNIT_ASSERT_VALUES_EQUAL(0, TEnumRange<TTestStruct::EMember4>::UnderlyingMin);
    UNIT_ASSERT_VALUES_EQUAL(2, TEnumRange<TTestStruct::EMember4>::UnderlyingMax);
}
