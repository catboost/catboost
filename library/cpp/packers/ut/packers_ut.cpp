#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/output.h>
#include <utility>

#include <util/charset/wide.h>
#include <util/generic/algorithm.h>
#include <util/generic/buffer.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/generic/ylimits.h>

#include <util/folder/dirut.h>

#include <util/random/random.h>

#include <util/string/hex.h>

#include "packers.h"

#include <array>
#include <iterator>

class TPackersTest: public TTestBase {
private:
    UNIT_TEST_SUITE(TPackersTest);
    UNIT_TEST(TestPackers);
    UNIT_TEST_SUITE_END();

    template <class TData, class TPacker>
    void TestPacker(const TData& data);

    template <class TData, class TPacker>
    void TestPacker(const TData* test, size_t size);

public:
    void TestPackers();
};

UNIT_TEST_SUITE_REGISTRATION(TPackersTest);

template <class TData, class TPacker>
void TPackersTest::TestPacker(const TData& data) {
    size_t len = TPacker().MeasureLeaf(data);
    size_t bufLen = len * 3;

    TArrayHolder<char> buf(new char[bufLen]);
    memset(buf.Get(), -1, bufLen);

    TPacker().PackLeaf(buf.Get(), data, len);

    UNIT_ASSERT(TPacker().SkipLeaf(buf.Get()) == len);

    TData dataTmp;
    TPacker().UnpackLeaf(buf.Get(), dataTmp);
    UNIT_ASSERT(data == dataTmp);
}

template <class TData, class TPacker>
void TPackersTest::TestPacker(const TData* test, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        TestPacker<TData, TPacker>(test[i]);
    }
}

void TPackersTest::TestPackers() {
    {
        const TString test[] = {"",
                                "a", "b", "c", "d",
                                "aa", "ab", "ac", "ad",
                                "aaa", "aab", "aac", "aad",
                                "aba", "abb", "abc", "abd",
                                "asdfjjmk.gjilsjgilsjilgjildsajgfilsjdfilgjm ldsa8oq43u 583uq4905 -q435 jiores u893q 5oiju fd-KE 89536 9Q2URE   12AI894T3 89 Q*(re43"};

        TestPacker<TString, NPackers::TPacker<TString>>(test, Y_ARRAY_SIZE(test));

        for (size_t i = 0; i != Y_ARRAY_SIZE(test); ++i) {
            TestPacker<TUtf16String, NPackers::TPacker<TUtf16String>>(UTF8ToWide(test[i]));
        }
    }
    {
        const ui64 test[] = {
            0, 1, 2, 3, 4, 5, 6, 76, 100000, Max<ui64>()};

        TestPacker<ui64, NPackers::TPacker<ui64>>(test, Y_ARRAY_SIZE(test));
    }
    {
        const int test[] = {
            0, 1, 2, 3, 4, 5, 6, 76, 100000, -1, -2, -3, -4, -5, -6, -76, -10000, Min<int>(), Max<int>()};

        TestPacker<int, NPackers::TPacker<int>>(test, Y_ARRAY_SIZE(test));
    }
    {
        const float test[] = {
            2.f, 3.f, 4.f, 0.f, -0.f, 1.f, -1.f, 1.1f, -1.1f,
            std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(),
            std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};

        TestPacker<float, NPackers::TFloatPacker>(test, Y_ARRAY_SIZE(test));
    }
    {
        const double test[] = {
            0., -0., 1., -1., 1.1, -1.1,
            std::numeric_limits<double>::min(), -std::numeric_limits<double>::min(),
            std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};

        TestPacker<double, NPackers::TDoublePacker>(test, Y_ARRAY_SIZE(test));
    }
}
