#include <catboost/libs/model/ut/lib/model_test_helpers.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/json/json_value.h>
#include <library/cpp/json/json_writer.h>

#include <util/stream/file.h>

using namespace std;
using namespace NCB;

namespace {
    TString BuildUserParams(const TString& Namespace) {
        NJson::TJsonValue params;
        params["namespace"] = Namespace;
        return NJson::WriteJson(params);
    }

    TString ReadFile(const TString& path) {
        return TFileInput(path).ReadAll();
    }
}

Y_UNIT_TEST_SUITE(TCppModelExportNamespace) {

    Y_UNIT_TEST(TestNoNamespace) {
        TFullModel model = TrainFloatCatboostModel();
        ExportModel(model, "model_no_ns.cpp", EModelType::CPP);

        TString code = ReadFile("model_no_ns.cpp");
        UNIT_ASSERT(!code.Contains("namespace"));
    }

    Y_UNIT_TEST(TestValidNamespace) {
        TFullModel model = TrainFloatCatboostModel();
        ExportModel(model, "model_valid_ns.cpp", EModelType::CPP, BuildUserParams("MyNamespace"));

        TString code = ReadFile("model_valid_ns.cpp");
        UNIT_ASSERT(code.Contains("namespace MyNamespace {"));
    }

    Y_UNIT_TEST(TestInvalidNamespace) {
        TFullModel model = TrainFloatCatboostModel();
        UNIT_ASSERT_EXCEPTION(
            ExportModel(model, "model_invalid_ns.cpp", EModelType::CPP, BuildUserParams("123abc")),
            yexception);
    }
}
