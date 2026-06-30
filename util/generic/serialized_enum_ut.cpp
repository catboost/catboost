#include "serialized_enum.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/deque.h>
#include <util/generic/map.h>
#include <util/generic/typelist.h>
#include <util/generic/vector.h>

Y_UNIT_TEST_SUITE(TestSerializedEnum) {
    Y_UNIT_TEST(RepresentationTypes) {
        using namespace NEnumSerializationRuntime::NDetail;

        static_assert(TIsPromotable<int, int>::value, "int -> int");
        static_assert(TIsPromotable<char, int>::value, "char -> int");
        static_assert(TIsPromotable<unsigned short, unsigned long>::value, "unsigned short -> unsigned long");
        static_assert(TIsPromotable<i64, long long>::value, "i64 -> long long");
        static_assert(!TIsPromotable<ui64, ui8>::value, "ui64 -> ui8");
        static_assert(!TIsPromotable<i64, short>::value, "i64 -> short");

        enum EEmpty {
        };
        UNIT_ASSERT_C((TTypeList<int, unsigned>::THave<typename TSelectEnumRepresentationType<EEmpty>::TType>::value), "empty enum using signed or unsigned integer underlying type");

        using TRepresentationTypeList = TTypeList<int, unsigned, long long, unsigned long long>;

        enum class ERegular {
            One = 1,
            Two = 2,
            Five = 5,
        };
        UNIT_ASSERT(TRepresentationTypeList::THave<typename TSelectEnumRepresentationType<ERegular>::TType>::value);

        enum class ESmall: unsigned char {
            Six = 6,
        };
        UNIT_ASSERT(TRepresentationTypeList::THave<typename TSelectEnumRepresentationType<ESmall>::TType>::value);

        enum class EHugeUnsigned: ui64 {
            Value = 0,
        };
        UNIT_ASSERT(TRepresentationTypeList::THave<typename TSelectEnumRepresentationType<EHugeUnsigned>::TType>::value);

        enum class EHugeSigned: i64 {
            Value = -2,
        };
        UNIT_ASSERT(TRepresentationTypeList::THave<typename TSelectEnumRepresentationType<EHugeSigned>::TType>::value);
    }

    Y_UNIT_TEST(MappedArrayView) {
        enum class ETestEnum: signed char {
            Zero = 0,
            One = 1,
            Two = 2,
            Three = 3,
            Four = 4,
            Eleven = 11,
        };
        const TVector<int> values = {1, 2, 3, 0, 0, 0, 11, 0, 0, 0, 0, 0, 2};
        const auto view = ::NEnumSerializationRuntime::TMappedArrayView<ETestEnum>{values};

        UNIT_ASSERT_VALUES_EQUAL(view.size(), values.size());
        UNIT_ASSERT_VALUES_EQUAL(view.empty(), false);
        UNIT_ASSERT_EQUAL(*view.begin(), ETestEnum::One);
        UNIT_ASSERT_EQUAL(view[6], ETestEnum::Eleven);
        UNIT_ASSERT_EXCEPTION(view.at(-1), std::out_of_range);
        UNIT_ASSERT_VALUES_EQUAL(sizeof(view[4]), sizeof(signed char));
        UNIT_ASSERT_VALUES_EQUAL(sizeof(values[4]), sizeof(int));
        for (const ETestEnum e : view) {
            UNIT_ASSERT_UNEQUAL(e, ETestEnum::Four);
        }

        const TVector<ETestEnum> typedValues = {ETestEnum::One, ETestEnum::Two, ETestEnum::Three, ETestEnum::Zero, ETestEnum::Zero, ETestEnum::Zero, ETestEnum::Eleven, ETestEnum::Zero, ETestEnum::Zero, ETestEnum::Zero, ETestEnum::Zero, ETestEnum::Zero, ETestEnum::Two};
        UNIT_ASSERT_EQUAL(typedValues, view.Materialize());

        const TDeque<ETestEnum> typedValuesDeque{typedValues.begin(), typedValues.end()};
        UNIT_ASSERT_EQUAL(typedValuesDeque, view.Materialize<TDeque>());
    }

    Y_UNIT_TEST(MappedDictView) {
        enum class ETestEnum: unsigned short {
            Zero = 0,
            One = 1,
            Two = 2,
            Three = 3,
            Four = 4,
            Eleven = 11,
            Fake = (unsigned short)(-1),
        };
        const TMap<unsigned, unsigned> map = {{0, 1}, {1, 2}, {2, 4}, {3, 8}, {4, 16}, {11, 2048}};
        const auto view = ::NEnumSerializationRuntime::NDetail::TMappedDictView<ETestEnum, unsigned, unsigned, decltype(map)>{map};

        UNIT_ASSERT_VALUES_EQUAL(view.size(), map.size());
        UNIT_ASSERT_VALUES_EQUAL(map.empty(), false);

        UNIT_ASSERT_EQUAL(view.begin()->first, ETestEnum::Zero);
        UNIT_ASSERT_VALUES_EQUAL(view.begin()->second, 1u);

        UNIT_ASSERT_VALUES_EQUAL(view.contains(ETestEnum::Fake), false);
        UNIT_ASSERT_VALUES_EQUAL(view.contains(ETestEnum::Four), true);

        UNIT_ASSERT_EXCEPTION(view.at(ETestEnum::Fake), std::out_of_range);
        UNIT_ASSERT_NO_EXCEPTION(view.at(ETestEnum::Eleven));

        UNIT_ASSERT_VALUES_EQUAL(view.at(ETestEnum::Three), 8u);

        unsigned mask = 0;
        unsigned sum = 0;
        for (const auto e : view) {
            mask |= e.second;
            sum += e.second;
        }
        UNIT_ASSERT_VALUES_EQUAL(mask, 2079);
        UNIT_ASSERT_VALUES_EQUAL(sum, 2079);

        const TMap<ETestEnum, unsigned> materialized = view.Materialize<TMap>();
        UNIT_ASSERT_VALUES_EQUAL(materialized.size(), map.size());
        UNIT_ASSERT_VALUES_EQUAL(materialized.at(ETestEnum::Four), 16);
    }
} // Y_UNIT_TEST_SUITE(TestSerializedEnum)
