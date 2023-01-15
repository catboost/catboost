#include "library/cpp/json/yson/json2yson.h"

#include <library/cpp/blockcodecs/codecs.h>
#include <library/cpp/histogram/simple/histogram.h>
#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

#include <util/datetime/cputimer.h>
#include <util/stream/file.h>

template <typename TCallBack>
ui64 Run(TCallBack&& callBack) {
    TSimpleTimer timer;
    callBack();
    return timer.Get().MicroSeconds();
}

static TString GetRequestsWithDecoding(const TString& inputPath, const NBlockCodecs::ICodec* codec) {
    TIFStream inputFileStream(inputPath);
    TString encodedRequests = inputFileStream.ReadAll();
    TString requests;
    codec->Decode(encodedRequests, requests);
    return requests;
}

Y_UNIT_TEST_SUITE(Json2Yson) {
    Y_UNIT_TEST(NOAPACHE_REQUESTS) {
        const ui32 warmUpRetries = 5;
        const TVector<double> percentiles = {0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0};

        NSimpleHistogram::TMultiHistogramCalcer<ui64> calcer;

        TString requests = GetRequestsWithDecoding(GetWorkPath() + "/noapache_requests_sample_lz4", NBlockCodecs::Codec("lz4"));
        TStringInput inputStream(requests);

        for (TString jsonRequest, jsonString, ysonString; inputStream.ReadLine(jsonRequest);) {
            TStringInput jsonInput(jsonRequest);
            NJson::TJsonValue readedJson;
            NJson::ReadJsonTree(&jsonInput, &readedJson, true);
            jsonRequest.clear();

            ui64 writeTime = Max<ui64>();
            ui64 readTime = Max<ui64>();

            for (ui32 i = 0; i < warmUpRetries; ++i) {
                NJson::TJsonValue Json2Json;
                TStringOutput jsonWriteOutput(jsonString);
                NJsonWriter::TBuf jsonBuf(NJsonWriter::HEM_UNSAFE, &jsonWriteOutput);

                writeTime = Min(writeTime, Run([&]() {
                                    jsonBuf.WriteJsonValue(&readedJson);
                                }));

                TStringInput jsonInput(jsonString);
                NJson::TJsonReaderConfig config;
                config.DontValidateUtf8 = true;
                readTime = Min(readTime, Run([&]() {
                                   NJson::ReadJsonTree(&jsonInput, &config, &Json2Json, true);
                               }));

                UNIT_ASSERT_VALUES_EQUAL(
                    NJsonWriter::TBuf().WriteJsonValue(&readedJson, true).Str(),
                    NJsonWriter::TBuf().WriteJsonValue(&Json2Json, true).Str());

                jsonString.clear();
            }

            calcer.RecordValue("read_json", readTime);
            calcer.RecordValue("write_json", writeTime);
            calcer.RecordValue("read_and_write_json", readTime + writeTime);

            writeTime = Max<ui64>();
            readTime = Max<ui64>();

            for (ui32 i = 0; i < warmUpRetries; ++i) {
                NJson::TJsonValue convertedJson;
                TStringOutput ysonOutput(ysonString);

                writeTime = Min(writeTime, Run([&]() {
                                    NJson2Yson::SerializeJsonValueAsYson(readedJson, &ysonOutput);
                                }));

                TStringInput ysonInput(ysonString);
                readTime = Min(readTime, Run([&]() {
                                   NJson2Yson::DeserializeYsonAsJsonValue(&ysonInput, &convertedJson);
                               }));

                UNIT_ASSERT_VALUES_EQUAL(
                    NJsonWriter::TBuf().WriteJsonValue(&convertedJson, true).Str(),
                    NJsonWriter::TBuf().WriteJsonValue(&readedJson, true).Str());

                ysonString.clear();
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
