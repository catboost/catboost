#include <library/cpp/testing/unittest/registar.h>

namespace NSubjectTests {
    class TAlwaysTearDownFixture : public NUnitTest::TBaseFixture {
    public:
        void TearDown(NUnitTest::TTestContext&) override {
            Cerr << Name_ << ": TearDown is ran" << Endl;
        }
    };

    class TAlwaysTearDownSetUpThrowsFixture : public NUnitTest::TBaseFixture {
    public:
        void SetUp(NUnitTest::TTestContext&) override {
            ythrow yexception() << "hope this won't skip teardown";
        }

        void TearDown(NUnitTest::TTestContext&) override {
            Cerr << Name_ << ": TearDown is ran" << Endl;
        }
    };

    Y_UNIT_TEST_SUITE(TestsAlwaysTearDown) {
        Y_UNIT_TEST_F(TestFail, TAlwaysTearDownFixture) {
            UNIT_ASSERT(false);
        }

        Y_UNIT_TEST_F(TestThrow, TAlwaysTearDownFixture) {
            ythrow yexception() << "hope this won't skip teardown";
        }

        Y_UNIT_TEST_F(TestSetUpThrows, TAlwaysTearDownSetUpThrowsFixture) {
        }
    }
}
