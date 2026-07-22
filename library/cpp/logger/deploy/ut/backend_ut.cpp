#include <library/cpp/logger/deploy/backend.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/logger/element.h>
#include <library/cpp/logger/log.h>
#include <library/cpp/logger/stream.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>

Y_UNIT_TEST_SUITE(DeployJsonLogBackendTest) {
    Y_UNIT_TEST(FormatBasicRecord) {
        const char* message = "hello";
        TLogRecord rec(TLOG_INFO, message, strlen(message));

        NJson::TJsonValue json;
        UNIT_ASSERT(NJson::ReadJsonTree(FormatDeployJsonLogRecord(rec, "test-logger"), &json));

        UNIT_ASSERT(json.Has("@timestamp"));
        UNIT_ASSERT_VALUES_EQUAL(json["levelStr"].GetStringSafe(), "INFO");
        UNIT_ASSERT(!json.Has("level"));
        UNIT_ASSERT_VALUES_EQUAL(json["message"].GetStringSafe(), "hello");
        UNIT_ASSERT_VALUES_EQUAL(json["loggerName"].GetStringSafe(), "test-logger");
        UNIT_ASSERT(!json.Has("@fields"));
        UNIT_ASSERT(!json.Has("threadName"));
    }

    Y_UNIT_TEST(LevelStrUsesEnumSerialization) {
        auto levelStrOf = [](ELogPriority priority) {
            const char* message = "x";
            TLogRecord rec(priority, message, 1);
            NJson::TJsonValue json;
            UNIT_ASSERT(NJson::ReadJsonTree(FormatDeployJsonLogRecord(rec), &json));
            return json["levelStr"].GetStringSafe();
        };

        UNIT_ASSERT_VALUES_EQUAL(levelStrOf(TLOG_EMERG), "EMERG");
        UNIT_ASSERT_VALUES_EQUAL(levelStrOf(TLOG_ALERT), "ALERT");
        UNIT_ASSERT_VALUES_EQUAL(levelStrOf(TLOG_CRIT), "CRITICAL_INFO");
        UNIT_ASSERT_VALUES_EQUAL(levelStrOf(TLOG_ERR), "ERROR");
        UNIT_ASSERT_VALUES_EQUAL(levelStrOf(TLOG_WARNING), "WARNING");
        UNIT_ASSERT_VALUES_EQUAL(levelStrOf(TLOG_NOTICE), "NOTICE");
        UNIT_ASSERT_VALUES_EQUAL(levelStrOf(TLOG_INFO), "INFO");
        UNIT_ASSERT_VALUES_EQUAL(levelStrOf(TLOG_DEBUG), "DEBUG");
        UNIT_ASSERT_VALUES_EQUAL(levelStrOf(TLOG_RESOURCES), "RESOURCES");
    }

    Y_UNIT_TEST(PromotesTopLevelMetaAndKeepsFields) {
        const char* message = "RequestStats";
        TLogRecord rec(
            TLOG_INFO,
            message,
            strlen(message),
            TLogRecord::TMetaFlags{
                {"request_id", "req-1"},
                {"trace.id", "trace-1"},
                {"code", "200"},
            });

        NJson::TJsonValue json;
        UNIT_ASSERT(NJson::ReadJsonTree(FormatDeployJsonLogRecord(rec), &json));

        UNIT_ASSERT_VALUES_EQUAL(json["request_id"].GetStringSafe(), "req-1");
        UNIT_ASSERT(!json.Has("trace.id"));
        UNIT_ASSERT_VALUES_EQUAL(json["@fields"]["trace.id"].GetStringSafe(), "trace-1");
        UNIT_ASSERT_VALUES_EQUAL(json["@fields"]["code"].GetStringSafe(), "200");
        UNIT_ASSERT(!json["@fields"].Has("request_id"));
    }

    Y_UNIT_TEST(LoggerNameIgnoresMetaFlags) {
        const char* message = "x";
        TLogRecord rec(
            TLOG_INFO,
            message,
            1,
            TLogRecord::TMetaFlags{{"loggerName", "from-meta"}});

        NJson::TJsonValue json;
        UNIT_ASSERT(NJson::ReadJsonTree(FormatDeployJsonLogRecord(rec, "from-backend"), &json));

        UNIT_ASSERT_VALUES_EQUAL(json["loggerName"].GetStringSafe(), "from-backend");
        UNIT_ASSERT(!json.Has("@fields"));
    }

    Y_UNIT_TEST(MetaFlagsLastWins) {
        const char* message = "dup";
        TLogRecord rec(
            TLOG_INFO,
            message,
            strlen(message),
            TLogRecord::TMetaFlags{
                {"file", "first.cpp"},
                {"request_id", "old"},
                {"file", "second.cpp"},
                {"request_id", "new"},
            });

        const TString formatted = FormatDeployJsonLogRecord(rec);
        NJson::TJsonValue json;
        UNIT_ASSERT(NJson::ReadJsonTree(formatted, &json));

        UNIT_ASSERT_VALUES_EQUAL(json["request_id"].GetStringSafe(), "new");
        UNIT_ASSERT_VALUES_EQUAL(json["@fields"]["file"].GetStringSafe(), "second.cpp");
        // NJson is last-wins too; assert the key was emitted once.
        UNIT_ASSERT_VALUES_EQUAL(formatted.find("\"file\""), formatted.rfind("\"file\""));
    }

    Y_UNIT_TEST(StripsTrailingNewlineFromMessage) {
        const char* message = "line\n";
        TLogRecord rec(TLOG_INFO, message, strlen(message));

        NJson::TJsonValue json;
        UNIT_ASSERT(NJson::ReadJsonTree(FormatDeployJsonLogRecord(rec), &json));
        UNIT_ASSERT_VALUES_EQUAL(json["message"].GetStringSafe(), "line");
    }

    Y_UNIT_TEST(WritesThroughSlaveBackend) {
        TStringStream output;
        TLog log(MakeHolder<TDeployJsonLogBackend>(
            MakeHolder<TStreamLogBackend>(&output),
            "suite"));

        {
            TLogElement element(&log, TLOG_WARNING);
            element.With("request_id", "req-1") << "warn";
        }

        NJson::TJsonValue json;
        UNIT_ASSERT(NJson::ReadJsonTree(output.Str(), &json));
        UNIT_ASSERT_VALUES_EQUAL(json["message"].GetStringSafe(), "warn");
        UNIT_ASSERT_VALUES_EQUAL(json["request_id"].GetStringSafe(), "req-1");
        UNIT_ASSERT_VALUES_EQUAL(json["loggerName"].GetStringSafe(), "suite");
    }
} // Y_UNIT_TEST_SUITE(DeployJsonLogBackendTest)
