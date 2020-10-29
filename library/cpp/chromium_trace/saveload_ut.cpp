#include "saveload.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/memory/pool.h>
#include <util/stream/str.h>

using namespace NChromiumTrace;

namespace {
    template <typename T>
    static TString SaveToString(const T& value) {
        TString data;
        TStringOutput out(data);
        ::Save(&out, value);
        return data;
    }

    template <typename T>
    static T LoadFromString(const TString& data, TMemoryPool& pool) {
        TStringInput in(data);
        T result;
        ::Load(&in, result, pool);
        return result;
    }

    template <typename T>
    static void TestSaveLoad(const T& value) {
        TMemoryPool pool(4096);
        TString data = SaveToString(value);
        T result = LoadFromString<T>(data, pool);
        UNIT_ASSERT_EQUAL(value, result);
    }
}

Y_UNIT_TEST_SUITE(SaveLoad) {
    Y_UNIT_TEST(EventArgs_Arg) {
        using TArg = TEventArgs::TArg;

        TestSaveLoad(TArg(TStringBuf("TestI64Arg"), i64(0xdeadbeef)));
        TestSaveLoad(TArg(TStringBuf("TestDoubleArg"), double(3.1415)));
        TestSaveLoad(TArg(TStringBuf("TestStringArg"), TStringBuf("Hello World!")));
    }

    Y_UNIT_TEST(EventArgs) {
        TestSaveLoad(TEventArgs());
        TestSaveLoad(TEventArgs()
                         .Add(TStringBuf("TestI64Arg"), i64(0xdeadbeef)));
        TestSaveLoad(TEventArgs()
                         .Add(TStringBuf("TestI64Arg"), i64(0xdeadbeef))
                         .Add(TStringBuf("TestDoubleArg"), double(3.1415))
                         .Add(TStringBuf("TestI64Arg"), TStringBuf("Hello World!")));
    }

    Y_UNIT_TEST(DurationBeginEvent) {
        TestSaveLoad(TDurationBeginEvent{
            TEventOrigin::Here(),
            "TestEvent",
            "Test,Sample",
            TEventTime::Now(),
            TEventFlow{
                EFlowType::Producer,
                0xdeadbeef,
            },
        });
    }

    Y_UNIT_TEST(DurationEndEvent) {
        TestSaveLoad(TDurationEndEvent{
            TEventOrigin::Here(),
            TEventTime::Now(),
            TEventFlow{
                EFlowType::Producer,
                0xdeadbeef,
            }});
    }

    Y_UNIT_TEST(DurationCompleteEvent) {
        TestSaveLoad(TDurationCompleteEvent{
            TEventOrigin::Here(),
            "TestEvent",
            "Test,Sample",
            TEventTime::Now(),
            TEventTime::Now(),
            TEventFlow{
                EFlowType::Producer,
                0xdeadbeef,
            }});
    }

    Y_UNIT_TEST(InstantEvent) {
        TestSaveLoad(TInstantEvent{
            TEventOrigin::Here(),
            "TestEvent",
            "Test,Sample",
            TEventTime::Now(),
            EScope::Process,
        });
    }

    Y_UNIT_TEST(AsyncEvent) {
        TestSaveLoad(TAsyncEvent{
            EAsyncEvent::Begin,
            TEventOrigin::Here(),
            "TestEvent",
            "Test,Sample",
            TEventTime::Now(),
            0xdeadbeef,
        });
    }

    Y_UNIT_TEST(CounterEvent) {
        TestSaveLoad(TCounterEvent{
            TEventOrigin::Here(),
            "TestEvent",
            "Test,Sample",
            TEventTime::Now(),
        });
    }

    Y_UNIT_TEST(MetadataEvent) {
        TestSaveLoad(TMetadataEvent{
            TEventOrigin::Here(),
            "TestEvent",
        });
    }

    Y_UNIT_TEST(EventWithArgs) {
        TestSaveLoad(TEventWithArgs{
            TCounterEvent{
                TEventOrigin::Here(),
                "TestEvent",
                "Test,Sample",
                TEventTime::Now(),
            },
        });
        TestSaveLoad(TEventWithArgs{
            TCounterEvent{
                TEventOrigin::Here(),
                "TestEvent",
                "Test,Sample",
                TEventTime::Now(),
            },
            TEventArgs()
                .Add("Int64Arg", i64(0xdeadbeef))});
        TestSaveLoad(TEventWithArgs{
            TMetadataEvent{
                TEventOrigin::Here(),
                "TestEvent",
            },
        });
        TestSaveLoad(TEventWithArgs{
            TMetadataEvent{
                TEventOrigin::Here(),
                "TestEvent",
            },
            TEventArgs()
                .Add("Int64Arg", i64(0xdeadbeef))});
    }

} // Y_UNIT_TEST_SUITE(SaveLoad)
