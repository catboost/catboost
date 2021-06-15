#include "gtest.h"
#include "simple.h"

#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <util/system/type_name.h>

using namespace NUnitTest;
using namespace NUnitTest::NPrivate;

IGTestFactory::~IGTestFactory() {
}

namespace {
    struct TCurrentTest: public TSimpleTestExecutor {
        inline TCurrentTest(TStringBuf name)
            : MyName(name)
        {
        }

        TString TypeId() const override {
            return TypeName(*this) + "-" + MyName;
        }

        TString Name() const noexcept override {
            return TString(MyName);
        }

        const TStringBuf MyName;
    };

    struct TGTestFactory: public IGTestFactory {
        inline TGTestFactory(TStringBuf name)
            : Test(name)
        {
        }

        ~TGTestFactory() override {
        }

        TString Name() const noexcept override {
            return Test.Name();
        }

        TTestBase* ConstructTest() override {
            return new TCurrentTest(Test);
        }

        void AddTest(const char* name, void (*body)(TTestContext&), bool forceFork) override {
            Test.Tests.push_back(TBaseTestCase(name, body, forceFork));
        }

        TCurrentTest Test;
    };
}

IGTestFactory* NUnitTest::NPrivate::ByName(const char* name) {
    static TMap<TStringBuf, TAutoPtr<TGTestFactory>> tests;

    auto& ret = tests[name];

    if (!ret) {
        ret = new TGTestFactory(name);
    }

    return ret.Get();
}
