#include "zerocopy_output.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/string.h>

// This version of string output stream is written here only
// for testing IZeroCopyOutput implementation of DoWrite.
class TSimpleStringOutput: public IZeroCopyOutput {
public:
    TSimpleStringOutput(TString& s) noexcept
        : S_(s)
    {
    }

private:
    size_t DoNext(void** ptr) override {
        if (S_.size() == S_.capacity()) {
            S_.reserve(FastClp2(S_.capacity() + 1));
        }
        size_t previousSize = S_.size();
        S_.resize(S_.capacity());
        *ptr = S_.begin() + previousSize;
        return S_.size() - previousSize;
    }

    void DoUndo(size_t len) override {
        Y_ENSURE(len <= S_.size());
        S_.resize(S_.size() - len);
    }

    TString& S_;
};

Y_UNIT_TEST_SUITE(TestZerocopyOutput) {
    Y_UNIT_TEST(Write) {
        TString str;
        TSimpleStringOutput output(str);
        TString result;

        for (size_t i = 0; i < 1000; ++i) {
            result.push_back('a' + (i % 20));
        }

        output.Write(result.begin(), result.size());
        UNIT_ASSERT_STRINGS_EQUAL(str, result);
    }
} // Y_UNIT_TEST_SUITE(TestZerocopyOutput)
