#include "aligned.h"

#include <library/cpp/testing/unittest/registar.h>

class TNastyInputStream: public IInputStream {
public:
    TNastyInputStream()
        : Pos_(0)
    {
    }

protected:
    size_t DoRead(void* buf, size_t len) override {
        if (len == 0) {
            return 0;
        }

        *static_cast<unsigned char*>(buf) = static_cast<unsigned char>(Pos_);
        ++Pos_;
        return 1;
    }

    size_t DoSkip(size_t len) override {
        if (len == 0) {
            return 0;
        }

        ++Pos_;
        return 1;
    }

private:
    size_t Pos_;
};

Y_UNIT_TEST_SUITE(TAlignedTest) {
    Y_UNIT_TEST(AlignInput) {
        TNastyInputStream input0;
        TAlignedInput alignedInput(&input0);

        char c = '\1';

        alignedInput.Align(2);
        alignedInput.ReadChar(c);
        UNIT_ASSERT_VALUES_EQUAL(c, '\x0');

        alignedInput.Align(2);
        alignedInput.ReadChar(c);
        UNIT_ASSERT_VALUES_EQUAL(c, '\x2');

        alignedInput.Align(4);
        alignedInput.ReadChar(c);
        UNIT_ASSERT_VALUES_EQUAL(c, '\x4');

        alignedInput.Align(16);
        alignedInput.ReadChar(c);
        UNIT_ASSERT_VALUES_EQUAL(c, '\x10');

        alignedInput.Align(128);
        alignedInput.ReadChar(c);
        UNIT_ASSERT_VALUES_EQUAL(c, '\x80');
    }
} // Y_UNIT_TEST_SUITE(TAlignedTest)
