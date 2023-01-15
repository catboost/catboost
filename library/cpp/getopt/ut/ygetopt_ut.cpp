#include <library/cpp/getopt/ygetopt.h>

#include <library/cpp/testing/unittest/registar.h>

class TGetOptTest: public TTestBase {
    UNIT_TEST_SUITE(TGetOptTest);
    UNIT_TEST(TestGetOpt);
    UNIT_TEST_EXCEPTION(TestZeroArgC, yexception);
    UNIT_TEST_SUITE_END();

public:
    void TestGetOpt();
    void TestZeroArgC();
};

UNIT_TEST_SUITE_REGISTRATION(TGetOptTest);

void TGetOptTest::TestZeroArgC() {
    TGetOpt opt(0, nullptr, "");
}

void TGetOptTest::TestGetOpt() {
    const char* argv[] = {
        "/usr/bin/bash",
        "-f",
        "-p",
        "qwerty123",
        "-z",
        "-q",
        nullptr};

    TString res;
    const TString format = "qzp:f";
    TGetOpt opt(sizeof(argv) / sizeof(*argv) - 1, argv, format);

    for (TGetOpt::TIterator it = opt.Begin(); it != opt.End(); ++it) {
        res += it->Key();

        if (it->HaveArg()) {
            res += it->Arg();
        }
    }

    UNIT_ASSERT_EQUAL(res, "fpqwerty123zq");
}
