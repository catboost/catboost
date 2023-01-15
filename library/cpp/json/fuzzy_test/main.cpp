#include <library/cpp/json/json_reader.h>

#include <util/random/random.h>
#include <util/stream/str.h>

extern "C" int LLVMFuzzerTestOneInput(const ui8* data, size_t size) {
    const auto json = TString((const char*)data, size);

    try {
        NJson::TJsonValue value;
        NJson::ReadJsonFastTree(json, &value, true);
    } catch (...) {
        //Cout << json << " -> " << CurrentExceptionMessage() << Endl;
    }

    try {
        NJson::TJsonCallbacks cb;
        NJson::ReadJsonFast(json, &cb);
    } catch (...) {
        //Cout << json << " -> " << CurrentExceptionMessage() << Endl;
    }

    try {
        NJson::ValidateJson(json);
    } catch (...) {
        //Cout << json << " -> " << CurrentExceptionMessage() << Endl;
    }

    return 0;
}
