#include "library/json/yson/json2yson.h"

#include <library/histogram/simple/histogram.h>
#include <library/unittest/registar.h>
#include <library/unittest/tests_data.h>

#include <util/datetime/cputimer.h>
#include <util/stream/file.h>

#include <web/app_host/lib/converter/converter.h>

template <typename TCallBack>
ui64 Run(TCallBack&& callBack) {
    TSimpleTimer timer;
    callBack();
    return timer.Get().MicroSeconds();
}

SIMPLE_UNIT_TEST_SUITE(Json2Yson) {
    SIMPLE_UNIT_TEST_WITH_CONTEXT(NOAPACHE_REQUESTS) {
        const ui32 warmUpRetries = 5;
        const yvector<double> percentiles = {0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0};

        const auto converter = NAppHost::NConverter::TConverterFactory().Create("service_request");
        NSimpleHistogram::TMultiHistogramCalcer<ui64> calcer;

        TIFStream inputStream(GetWorkPath() + "/noapache_requests_sample");
        for (TString convertedRequest, buffer; inputStream.ReadLine(buffer);) {
            convertedRequest = converter->ConvertToJSON(buffer);
            TStringInput jsonInput(convertedRequest);
            NJson::TJsonValue readedJson;
            NJson::ReadJsonTree(&jsonInput, &readedJson, true);
            buffer.clear();

            ui64 writeTime = Max<ui64>();
            ui64 readTime = Max<ui64>();

            for (ui32 i = 0; i < warmUpRetries; ++i) {
                NJson::TJsonValue Json2Json;
                TStringOutput jsonWriteOutput(buffer);
                NJsonWriter::TBuf jsonBuf(NJsonWriter::HEM_UNSAFE, &jsonWriteOutput);

                writeTime = Min(writeTime, Run([&]() {
                                    jsonBuf.WriteJsonValue(&readedJson);
                                }));

                TStringInput jsonInput(buffer);
                NJson::TJsonReaderConfig config;
                config.DontValidateUtf8 = true;
                readTime = Min(readTime, Run([&]() {
                                   NJson::ReadJsonTree(&jsonInput, &config, &Json2Json, true);
                               }));

                UNIT_ASSERT_VALUES_EQUAL(
                    NJsonWriter::TBuf().WriteJsonValue(&readedJson, true).Str(),
                    NJsonWriter::TBuf().WriteJsonValue(&Json2Json, true).Str());

                buffer.clear();
            }

            calcer.RecordValue("read_json", readTime);
            calcer.RecordValue("write_json", writeTime);
            calcer.RecordValue("read_and_write_json", readTime + writeTime);

            writeTime = Max<ui64>();
            readTime = Max<ui64>();

            for (ui32 i = 0; i < warmUpRetries; ++i) {
                NJson::TJsonValue convertedJson;
                TStringOutput ysonOutput(buffer);

                writeTime = Min(writeTime, Run([&]() {
                                    NJson2Yson::ConvertJson2Yson(readedJson, &ysonOutput);
                                }));

                TStringInput ysonInput(buffer);
                readTime = Min(readTime, Run([&]() {
                                   NJson2Yson::ConvertYson2Json(&ysonInput, &convertedJson);
                               }));

                UNIT_ASSERT_VALUES_EQUAL(
                    NJsonWriter::TBuf().WriteJsonValue(&convertedJson, true).Str(),
                    NJsonWriter::TBuf().WriteJsonValue(&readedJson, true).Str());

                buffer.clear();
            }

            calcer.RecordValue("read_yson", readTime);
            calcer.RecordValue("write_yson", writeTime);
            calcer.RecordValue("read_and_write_yson", readTime + writeTime);
        }

        NJson::TJsonValue histogramJson = NSimpleHistogram::ToJson(calcer.Calc(), percentiles);
        for (const auto& it : histogramJson.GetMap()) {
            for (const auto& percentileValue : it.second.GetMap()) {
                UNIT_ADD_METRIC(it.first + "_" + percentileValue.first, percentileValue.second.GetUInteger() / 1000.0);
            }
        }
    }
}
