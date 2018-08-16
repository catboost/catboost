#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <util/generic/string.h>

#include <library/unittest/registar.h>

struct TStringHolder : public NCB::IResourceHolder {
    TString S;
};


Y_UNIT_TEST_SUITE(TMaybeOwningArrayHolder) {
    Y_UNIT_TEST(TestNonOwning) {
        TVector<int> v{1, 2, 3};

        auto arrayHolder =  NCB::TMaybeOwningArrayHolder<int>::CreateNonOwning(v);

        UNIT_ASSERT_EQUAL(*arrayHolder, TArrayRef<int>(v));
    }

    Y_UNIT_TEST(TestGenericOwning) {
        auto stringHolder = MakeIntrusive<TStringHolder>();
        stringHolder->S = "string";

        auto arrayHolder = NCB::TMaybeOwningArrayHolder<const char>::CreateOwning(
            TConstArrayRef<char>(stringHolder->S.cbegin(), stringHolder->S.cend()),
            stringHolder
        );

        UNIT_ASSERT_EQUAL(*arrayHolder, (TConstArrayRef<char>({'s', 't', 'r', 'i', 'n', 'g'})));
    }

    Y_UNIT_TEST(TestVectorOwning) {
        TVector<TString> v{"aa", "bbb", "cccc", "d"};

        auto arrayHolder =  NCB::TMaybeOwningArrayHolder<TString>::CreateOwning(std::move(v));

        UNIT_ASSERT_EQUAL(*arrayHolder, (TArrayRef<TString>)(TVector<TString>{"aa", "bbb", "cccc", "d"}));
    }
}
