#include <library/unittest/registar.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/serialized_enum.h>
#include <util/generic/vector.h>

static const TConstArrayRef<size_t> MAX_BORDER_COUNT_VALUES = {1, 10, 128};
static const TConstArrayRef<bool> NAN_IS_INFINITY_VALUES = {true, false};

template <typename C, typename F>
static void ForEachValue(const C& values, F&& foo) {
    for (const auto& v : values) {
        foo(v);
    }
}

template <typename E, typename F>
static void ForEachEnumValue(F&& foo) {
    ForEachValue(GetEnumAllValues<E>(), foo);
}

Y_UNIT_TEST_SUITE(BinarizationTests) {
    Y_UNIT_TEST(TestEmpty) {
        ForEachEnumValue<EBorderSelectionType>([](const auto borderSelectionAlgorithm) {
            ForEachValue(NAN_IS_INFINITY_VALUES, [&](const auto nanIsInfinity) {
                ForEachValue(MAX_BORDER_COUNT_VALUES, [&](const auto maxBorderCount) {
                    TVector<float> values;
                    BestSplit(values, maxBorderCount, borderSelectionAlgorithm, nanIsInfinity);
                });
            });
        });
    }
}
