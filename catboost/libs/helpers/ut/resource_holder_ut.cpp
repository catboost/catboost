#include <catboost/libs/helpers/resource_holder.h>

#include <util/generic/string.h>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(TResourceHolder) {
    Y_UNIT_TEST(TestVectorHolder) {
        {
            NCB::TVectorHolder<int> vh;
            UNIT_ASSERT(vh.Data.empty());
        }

        {
            TIntrusivePtr<NCB::IResourceHolder> resourceHolder =
                MakeIntrusive<NCB::TVectorHolder<TString>>(TVector<TString>{{"a", "bb", "ccc", "d"}});

            auto& vh = dynamic_cast<NCB::TVectorHolder<TString>&>(*resourceHolder);
            UNIT_ASSERT_EQUAL(vh.Data, (TVector<TString>{"a", "bb", "ccc", "d"}));
        }

        {
            TVector<int> v{{1,2,3}};
            TIntrusivePtr<NCB::IResourceHolder> resourceHolder =
                MakeIntrusive<NCB::TVectorHolder<int>>(std::move(v));

            auto& vh = dynamic_cast<NCB::TVectorHolder<int>&>(*resourceHolder);
            UNIT_ASSERT_EQUAL(vh.Data, (TVector<int>{1, 2, 3}));
        }
    }
}
