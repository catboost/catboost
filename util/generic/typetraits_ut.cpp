#include "typetraits.h"

#include <library/cpp/testing/unittest/registar.h>

#include <vector>
#include <tuple>

namespace {
    enum ETestEnum {
    };

    class TPodClass {
    };

    class TNonPodClass {
        TNonPodClass() {
        }
    };

    class TEmptyClass {
        void operator()() const {
        }
    };

    class TAnotherEmptyClass {
    };

    class TEmptyDerivedClass: public TEmptyClass {
    };

    class TEmptyMultiDerivedClass: public TEmptyDerivedClass, public TAnotherEmptyClass {
        /* Not empty under MSVC.
         * MSVC's EBCO implementation can handle only one empty base class. */
    };

    struct TNonEmptyClass {
        TEmptyClass member;
    };

    class TNonEmptyDerivedClass: public TNonEmptyClass {
    };

    class TStdLayoutClass1: public TEmptyClass {
    public:
        int Value1;
        int Value2;
    };

    class TStdLayoutClass2: public TNonEmptyClass {
    };

    class TNonStdLayoutClass1 {
    public:
        int Value1;

    protected:
        int Value2;
    };

    class TNonStdLayoutClass2 {
    public:
        virtual void Func() {
        }
    };

    class TNonStdLayoutClass3: public TNonStdLayoutClass2 {
    };

    class TNonStdLayoutClass4: public TEmptyClass {
    public:
        TEmptyClass Base;
    };
}

#define ASSERT_SAME_TYPE(x, y)                     \
    {                                              \
        const bool x_ = std::is_same<x, y>::value; \
        UNIT_ASSERT_C(x_, #x " != " #y);           \
    }

Y_UNIT_TEST_SUITE(TTypeTraitsTest) {
    Y_UNIT_TEST(TestIsSame) {
        UNIT_ASSERT((std::is_same<int, int>::value));
        UNIT_ASSERT(!(std::is_same<signed int, unsigned int>::value));
    }

    Y_UNIT_TEST(TestRemoveReference) {
        ASSERT_SAME_TYPE(std::remove_reference_t<int>, int);
        ASSERT_SAME_TYPE(std::remove_reference_t<const int>, const int);
        ASSERT_SAME_TYPE(std::remove_reference_t<int&>, int);
        ASSERT_SAME_TYPE(std::remove_reference_t<const int&>, const int);
        ASSERT_SAME_TYPE(std::remove_reference_t<int&&>, int);
        ASSERT_SAME_TYPE(std::remove_reference_t<const int&&>, const int);

        class TIncompleteType;
        ASSERT_SAME_TYPE(std::remove_reference_t<TIncompleteType&>, TIncompleteType);
    }

    Y_UNIT_TEST(TestRemoveConst) {
        ASSERT_SAME_TYPE(std::remove_const_t<const int>, int);
    }

    Y_UNIT_TEST(TestRemoveVolatile) {
        ASSERT_SAME_TYPE(std::remove_volatile_t<volatile int>, int);
    }

    Y_UNIT_TEST(TestRemoveCV) {
        ASSERT_SAME_TYPE(std::remove_cv_t<const volatile int>, int);
    }

    Y_UNIT_TEST(TestAddCV) {
        ASSERT_SAME_TYPE(std::add_cv_t<int>, const volatile int);
    }

    Y_UNIT_TEST(TestClass) {
        UNIT_ASSERT(std::is_class<TString>::value);
        UNIT_ASSERT(!std::is_class<ETestEnum>::value);
        UNIT_ASSERT(!std::is_class<int>::value);
        UNIT_ASSERT(!std::is_class<void*>::value);
    }

    template <class T>
    inline void TestArithmeticType() {
        UNIT_ASSERT(std::is_arithmetic<T>::value);
        UNIT_ASSERT(std::is_arithmetic<const T>::value);
        UNIT_ASSERT(std::is_arithmetic<volatile T>::value);
        UNIT_ASSERT(std::is_arithmetic<const volatile T>::value);

        UNIT_ASSERT(!std::is_arithmetic<T&>::value);
        UNIT_ASSERT(!std::is_arithmetic<T&&>::value);
        UNIT_ASSERT(!std::is_arithmetic<T*>::value);

        bool a;

        a = std::is_same<typename TTypeTraits<T>::TFuncParam, T>::value;
        UNIT_ASSERT(a);
        a = std::is_same<typename TTypeTraits<const volatile T>::TFuncParam, const volatile T>::value;
        UNIT_ASSERT(a);
    }

    template <class T>
    inline void TestUnsignedIntType() {
        UNIT_ASSERT(std::is_unsigned<T>::value);
        UNIT_ASSERT(std::is_unsigned<const T>::value);
        UNIT_ASSERT(std::is_unsigned<volatile T>::value);
        UNIT_ASSERT(std::is_unsigned<const volatile T>::value);

        UNIT_ASSERT(!std::is_unsigned<T&>::value);
        UNIT_ASSERT(!std::is_unsigned<T&&>::value);
        UNIT_ASSERT(!std::is_unsigned<T*>::value);

        enum ETypedEnum: T {};
        UNIT_ASSERT(!std::is_unsigned<ETypedEnum>::value);
    }

    template <class T>
    inline void TestSignedIntType() {
        UNIT_ASSERT(std::is_signed<T>::value);
        UNIT_ASSERT(std::is_signed<const T>::value);
        UNIT_ASSERT(std::is_signed<volatile T>::value);
        UNIT_ASSERT(std::is_signed<const volatile T>::value);

        UNIT_ASSERT(!std::is_signed<T&>::value);
        UNIT_ASSERT(!std::is_signed<T&&>::value);
        UNIT_ASSERT(!std::is_signed<T*>::value);

        enum ETypedEnum: T {};
        UNIT_ASSERT(!std::is_signed<ETypedEnum>::value);
    }

    Y_UNIT_TEST(TestBool) {
        TestArithmeticType<bool>();
        TestUnsignedIntType<bool>();
    }

    Y_UNIT_TEST(TestUnsignedChar) {
        TestArithmeticType<unsigned char>();
        TestUnsignedIntType<unsigned char>();
    }

    Y_UNIT_TEST(TestSizeT) {
        TestArithmeticType<size_t>();
        TestUnsignedIntType<size_t>();
    }

    Y_UNIT_TEST(TestInt) {
        TestArithmeticType<int>();
        TestSignedIntType<int>();
    }

    Y_UNIT_TEST(TestDouble) {
        TestArithmeticType<double>();
    }

    Y_UNIT_TEST(TestLongDouble) {
        TestArithmeticType<long double>();
    }

    Y_UNIT_TEST(TestAddRValueReference) {
        ASSERT_SAME_TYPE(std::add_rvalue_reference_t<int>, int&&);
        ASSERT_SAME_TYPE(std::add_rvalue_reference_t<int const&>, int const&);
        ASSERT_SAME_TYPE(std::add_rvalue_reference_t<int*>, int*&&);
        ASSERT_SAME_TYPE(std::add_rvalue_reference_t<int*&>, int*&);
        ASSERT_SAME_TYPE(std::add_rvalue_reference_t<int&&>, int&&);
        ASSERT_SAME_TYPE(std::add_rvalue_reference_t<void>, void);
    }

    Y_UNIT_TEST(TestIsEmpty) {
        UNIT_ASSERT(std::is_empty<TEmptyClass>::value);
        UNIT_ASSERT(std::is_empty<TEmptyDerivedClass>::value);
        UNIT_ASSERT(std::is_empty<TAnotherEmptyClass>::value);
#ifdef _MSC_VER
        UNIT_ASSERT(!std::is_empty<TEmptyMultiDerivedClass>::value);
#else
        UNIT_ASSERT(std::is_empty<TEmptyMultiDerivedClass>::value);
#endif
        UNIT_ASSERT(!std::is_empty<TNonEmptyClass>::value);
        UNIT_ASSERT(!std::is_empty<TNonEmptyDerivedClass>::value);
    }

    Y_UNIT_TEST(TestIsStandardLayout) {
        UNIT_ASSERT(std::is_standard_layout<TStdLayoutClass1>::value);
        UNIT_ASSERT(std::is_standard_layout<TStdLayoutClass2>::value);
        UNIT_ASSERT(!std::is_standard_layout<TNonStdLayoutClass1>::value);
        UNIT_ASSERT(!std::is_standard_layout<TNonStdLayoutClass2>::value);
        UNIT_ASSERT(!std::is_standard_layout<TNonStdLayoutClass3>::value);
        UNIT_ASSERT(!std::is_standard_layout<TNonStdLayoutClass4>::value);
    }

    template <class T>
    using TTrySum = decltype(std::declval<T>() + std::declval<T>());

    Y_UNIT_TEST(TestIsTriviallyCopyable) {
        struct TPod {
            int value;
        };

        struct TNontriviallyCopyAssignable {
            TNontriviallyCopyAssignable(const TNontriviallyCopyAssignable&) = default;
            TNontriviallyCopyAssignable& operator=(const TNontriviallyCopyAssignable&);
        };

        struct TNonTriviallyCopyConstructible {
            TNonTriviallyCopyConstructible(const TNonTriviallyCopyConstructible&);
            TNonTriviallyCopyConstructible& operator=(const TNonTriviallyCopyConstructible&) = default;
        };

        struct TNonTriviallyDestructible {
            TNonTriviallyDestructible(const TNonTriviallyDestructible&) = default;
            TNonTriviallyDestructible& operator=(const TNonTriviallyDestructible&) = default;
            ~TNonTriviallyDestructible();
        };

        UNIT_ASSERT(std::is_trivially_copyable<int>::value);
        UNIT_ASSERT(std::is_trivially_copyable<TPod>::value);
        UNIT_ASSERT(!std::is_trivially_copyable<TNontriviallyCopyAssignable>::value);
        UNIT_ASSERT(!std::is_trivially_copyable<TNonTriviallyCopyConstructible>::value);
        UNIT_ASSERT(!std::is_trivially_copyable<TNonTriviallyDestructible>::value);
    }
}

namespace {
    template <typename T>
    struct TTypeTraitsExpected;

    template <>
    struct TTypeTraitsExpected<void> {
        enum { IsIntegral = false };
        enum { IsArithmetic = false };
        enum { IsPod = true };
        enum { IsVolatile = false };
        enum { IsConstant = false };
        enum { IsPointer = false };
        enum { IsReference = false };
        enum { IsLvalueReference = false };
        enum { IsRvalueReference = false };
        enum { IsArray = false };
        enum { IsClassType = false };
        enum { IsVoid = true };
        enum { IsEnum = false };
    };

    template <>
    struct TTypeTraitsExpected<int>: public TTypeTraitsExpected<void> {
        enum { IsIntegral = true };
        enum { IsArithmetic = true };
        enum { IsVoid = false };
    };

    template <>
    struct TTypeTraitsExpected<size_t>: public TTypeTraitsExpected<int> {
    };

    template <>
    struct TTypeTraitsExpected<float>: public TTypeTraitsExpected<int> {
        enum { IsIntegral = false };
    };

    template <>
    struct TTypeTraitsExpected<long double>: public TTypeTraitsExpected<float> {
    };

    template <>
    struct TTypeTraitsExpected<const int>: public TTypeTraitsExpected<int> {
        enum { IsConstant = true };
    };

    template <>
    struct TTypeTraitsExpected<volatile int>: public TTypeTraitsExpected<int> {
        enum { IsVolatile = true };
    };

    template <>
    struct TTypeTraitsExpected<ETestEnum>: public TTypeTraitsExpected<int> {
        enum { IsIntegral = false };
        enum { IsArithmetic = false };
        enum { IsEnum = true };
    };

    template <>
    struct TTypeTraitsExpected<TPodClass>: public TTypeTraitsExpected<void> {
        enum { IsClassType = true };
        enum { IsVoid = false };
    };

    template <>
    struct TTypeTraitsExpected<TNonPodClass>: public TTypeTraitsExpected<TPodClass> {
        enum { IsPod = false };
    };

    template <>
    struct TTypeTraitsExpected<TNonPodClass&>: public TTypeTraitsExpected<TNonPodClass> {
        enum { IsClassType = false };
        enum { IsReference = true };
        enum { IsLvalueReference = true };
    };

    template <>
    struct TTypeTraitsExpected<TNonPodClass&&>: public TTypeTraitsExpected<TNonPodClass> {
        enum { IsClassType = false };
        enum { IsReference = true };
        enum { IsRvalueReference = true };
    };

    template <>
    struct TTypeTraitsExpected<const TNonPodClass&>: public TTypeTraitsExpected<TNonPodClass&> {
    };

    template <>
    struct TTypeTraitsExpected<float*>: public TTypeTraitsExpected<int> {
        enum { IsIntegral = false };
        enum { IsArithmetic = false };
        enum { IsPointer = true };
    };

    template <>
    struct TTypeTraitsExpected<float&>: public TTypeTraitsExpected<float*> {
        enum { IsPointer = false };
        enum { IsReference = true };
        enum { IsLvalueReference = true };
    };

    template <>
    struct TTypeTraitsExpected<float&&>: public TTypeTraitsExpected<float*> {
        enum { IsPointer = false };
        enum { IsReference = true };
        enum { IsRvalueReference = true };
    };

    template <>
    struct TTypeTraitsExpected<const float&>: public TTypeTraitsExpected<float&> {
    };

    template <>
    struct TTypeTraitsExpected<float[17]>: public TTypeTraitsExpected<int> {
        enum { IsIntegral = false };
        enum { IsArithmetic = false };
        enum { IsArray = true };
    };
}

#define UNIT_ASSERT_EQUAL_ENUM(expected, actual) UNIT_ASSERT_VALUES_EQUAL((bool)(expected), (bool)(actual))

Y_UNIT_TEST_SUITE(TTypeTraitsTestNg) {
    template <typename T>
    void TestImpl() {
        // UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsPod, TTypeTraits<T>::IsPod);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsVoid, std::is_void<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsEnum, std::is_enum<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsIntegral, std::is_integral<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsArithmetic, std::is_arithmetic<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsVolatile, std::is_volatile<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsConstant, std::is_const<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsPointer, std::is_pointer<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsReference, std::is_reference<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsLvalueReference, std::is_lvalue_reference<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsRvalueReference, std::is_rvalue_reference<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsArray, std::is_array<T>::value);
        UNIT_ASSERT_EQUAL_ENUM(TTypeTraitsExpected<T>::IsClassType, std::is_class<T>::value);
    }

#define TYPE_TEST(name, type) \
    Y_UNIT_TEST(name) {       \
        TestImpl<type>();     \
    }

    TYPE_TEST(Void, void)
    TYPE_TEST(Int, int)
    TYPE_TEST(Float, float)
    TYPE_TEST(LongDouble, long double)
    TYPE_TEST(SizeT, size_t)
    TYPE_TEST(VolatileInt, volatile int)
    TYPE_TEST(ConstInt, const int)
    TYPE_TEST(Enum, ETestEnum)
    TYPE_TEST(FloatPointer, float*)
    TYPE_TEST(FloatReference, float&)
    TYPE_TEST(FloatConstReference, const float&)
    TYPE_TEST(FloatArray, float[17])
    TYPE_TEST(PodClass, TPodClass)
    TYPE_TEST(NonPodClass, TNonPodClass)
    TYPE_TEST(NonPodClassReference, TNonPodClass&)
    TYPE_TEST(NonPodClassConstReference, const TNonPodClass&)
}

enum E4 {
    X
};

enum class E64: ui64 {
    X
};

enum class E8: ui8 {
    X
};

// test for std::underlying_type_t
static_assert(sizeof(std::underlying_type_t<E4>) == sizeof(int), "");
static_assert(sizeof(std::underlying_type_t<E64>) == sizeof(ui64), "");
static_assert(sizeof(std::underlying_type_t<E8>) == sizeof(ui8), "");

// tests for TFixedWidthUnsignedInt
static_assert(std::is_same<ui8, TFixedWidthUnsignedInt<i8>>::value, "");
static_assert(std::is_same<ui16, TFixedWidthUnsignedInt<i16>>::value, "");
static_assert(std::is_same<ui32, TFixedWidthUnsignedInt<i32>>::value, "");
static_assert(std::is_same<ui64, TFixedWidthUnsignedInt<i64>>::value, "");

// tests for TFixedWidthSignedInt
static_assert(std::is_same<i8, TFixedWidthSignedInt<ui8>>::value, "");
static_assert(std::is_same<i16, TFixedWidthSignedInt<ui16>>::value, "");
static_assert(std::is_same<i32, TFixedWidthSignedInt<ui32>>::value, "");
static_assert(std::is_same<i64, TFixedWidthSignedInt<ui64>>::value, "");

// test for TIsSpecializationOf
static_assert(TIsSpecializationOf<std::vector, std::vector<int>>::value, "");
static_assert(TIsSpecializationOf<std::tuple, std::tuple<int, double, char>>::value, "");
static_assert(!TIsSpecializationOf<std::vector, std::tuple<int, double, char>>::value, "");
static_assert(!TIsSpecializationOf<std::pair, std::vector<int>>::value, "");

// test for TIsTemplateBaseOf
static_assert(TIsTemplateBaseOf<std::vector, std::vector<int>>::value);
static_assert(TIsTemplateBaseOf<std::tuple, std::tuple<int, double, char>>::value);
static_assert(TIsTemplateBaseOf<std::basic_string_view, std::wstring_view>::value);
static_assert(TIsTemplateBaseOf<std::vector, TVector<int>>::value);
static_assert(!TIsTemplateBaseOf<TVector, std::vector<int>>::value);
static_assert(TIsTemplateBaseOf<TBasicStringBuf, TWtringBuf>::value);
static_assert(TIsTemplateBaseOf<std::basic_string_view, TUtf32StringBuf>::value);
static_assert(TIsTemplateBaseOf<std::basic_string_view, TWtringBuf>::value);

// test for TIsIterable
static_assert(TIsIterable<std::vector<int>>::value, "");
static_assert(!TIsIterable<int>::value, "");
static_assert(TIsIterable<int[42]>::value, "");

// test for TDependentFalse
static_assert(TDependentFalse<int> == false);
static_assert(TDependentFalse<TNonPodClass> == false);
static_assert(TValueDependentFalse<0x1000> == false);
