#include "walk.h"

#include <library/unittest/registar.h>

class TStringListInput: public IWalkInput {
public:
    TStringListInput(const yvector<TString>& data)
        : Data_(data)
        , Index_(0)
    {
    }

protected:
    size_t DoUnboundedNext(const void** ptr) override {
        if (Index_ >= Data_.size()) {
            return 0;
        }

        const TString& string = Data_[Index_++];

        *ptr = string.data();
        return string.size();
    }

private:
    const yvector<TString>& Data_;
    size_t Index_;
};

SIMPLE_UNIT_TEST_SUITE(TWalkTest) {
    SIMPLE_UNIT_TEST(ReadTo) {
        yvector<TString> data;
        data.push_back("111a");
        data.push_back("222b");
        data.push_back("333c");
        data.push_back("444d");
        data.push_back("555e");
        data.push_back("666f");

        TStringListInput input(data);

        TString tmp1 = input.ReadTo('c');
        UNIT_ASSERT_VALUES_EQUAL(tmp1, "111a222b333");

        char tmp2;
        input.Read(&tmp2, 1);
        UNIT_ASSERT_VALUES_EQUAL(tmp2, '4');

        TString tmp3 = input.ReadTo('6');
        UNIT_ASSERT_VALUES_EQUAL(tmp3, "44d555e");

        TString tmp4 = input.ReadAll();
        UNIT_ASSERT_VALUES_EQUAL(tmp4, "66f");
    }
}
