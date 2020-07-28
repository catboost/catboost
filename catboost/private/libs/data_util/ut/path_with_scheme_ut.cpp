#include <library/cpp/testing/unittest/registar.h>

#include <catboost/private/libs/data_util/path_with_scheme.h>


using namespace NCB;


struct ITestProcessor {
    virtual TString Name() const = 0;
    virtual ~ITestProcessor() = default;
};

struct TFileProcessor : public ITestProcessor {
    static TString GetTypeName() { return ""; }

    TString Name() const override { return "File"; }
};

struct TYTProcessor : public ITestProcessor {
    static TString GetTypeName() { return "yt"; }

    TString Name() const override { return "YT"; }
};



using TTestFactory = NObjectFactory::TParametrizedObjectFactory<ITestProcessor, TString>;

static TTestFactory::TRegistrator<TFileProcessor> FileProcessorReg;
static TTestFactory::TRegistrator<TYTProcessor> YTProcessorReg;

Y_UNIT_TEST_SUITE(TPathWithOptionsTest) {
    Y_UNIT_TEST(TestCtor) {
        {
            TPathWithScheme p1("file1");
            UNIT_ASSERT_EQUAL(p1.Scheme, "");
            UNIT_ASSERT_EQUAL(p1.Path, "file1");
        }

        {
            TPathWithScheme p1("yt://hahn/path2");
            UNIT_ASSERT_EQUAL(p1.Scheme, "yt");
            UNIT_ASSERT_EQUAL(p1.Path, "hahn/path2");
        }

        UNIT_ASSERT_EXCEPTION(TPathWithScheme(""), yexception);
        UNIT_ASSERT_EXCEPTION(TPathWithScheme("://"), yexception);
        UNIT_ASSERT_EXCEPTION(TPathWithScheme("://path1"), yexception);
        UNIT_ASSERT_EXCEPTION(TPathWithScheme("scheme1://"), yexception);
    }

    Y_UNIT_TEST(TestGetProcessor) {
        {
            THolder<ITestProcessor> fileProc(GetProcessor<ITestProcessor>(TPathWithScheme("file")));
            UNIT_ASSERT_EQUAL(fileProc->Name(), "File");
        }
        {
            THolder<ITestProcessor> ytProc(
                GetProcessor<ITestProcessor>(TPathWithScheme("yt://hahn/path"))
            );
            UNIT_ASSERT_EQUAL(ytProc->Name(), "YT");
        }
        {
            UNIT_ASSERT_EXCEPTION(
                GetProcessor<ITestProcessor>(TPathWithScheme("unk://path2")),
                yexception
            );
        }
    }
}
