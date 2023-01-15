#pragma once

#include "consumer.h"
#include "event.h"

#include <util/ysaveload.h>
#include <util/memory/pool.h>

namespace NChromiumTrace {
    class TSaveLoadTraceConsumer final: public ITraceConsumer {
        IOutputStream* Stream;

    public:
        TSaveLoadTraceConsumer(IOutputStream* stream);

        void AddEvent(const TDurationBeginEvent& event, const TEventArgs* args) override;
        void AddEvent(const TDurationEndEvent& event, const TEventArgs* args) override;
        void AddEvent(const TDurationCompleteEvent& event, const TEventArgs* args) override;
        void AddEvent(const TCounterEvent& event, const TEventArgs* args) override;
        void AddEvent(const TMetadataEvent& event, const TEventArgs* args) override;
    };
}

template <>
class TSerializer<NChromiumTrace::TEventArgs::TArg> {
public:
    static void Save(IOutputStream* out, const NChromiumTrace::TEventArgs::TArg& value);
    static void Load(IInputStream* in, NChromiumTrace::TEventArgs::TArg& value, TMemoryPool& pool);
};

template <>
class TSerializer<NChromiumTrace::TEventArgs> {
public:
    static void Save(IOutputStream* out, const NChromiumTrace::TEventArgs& value);
    static void Load(IInputStream* in, NChromiumTrace::TEventArgs& value, TMemoryPool& pool);
};

template <>
class TSerializer<NChromiumTrace::TDurationBeginEvent> {
public:
    static void Save(IOutputStream* out, const NChromiumTrace::TDurationBeginEvent& value);
    static void Load(IInputStream* in, NChromiumTrace::TDurationBeginEvent& value, TMemoryPool& pool);
};

template <>
class TSerializer<NChromiumTrace::TDurationCompleteEvent> {
public:
    static void Save(IOutputStream* out, const NChromiumTrace::TDurationCompleteEvent& value);
    static void Load(IInputStream* in, NChromiumTrace::TDurationCompleteEvent& value, TMemoryPool& pool);
};

template <>
class TSerializer<NChromiumTrace::TInstantEvent> {
public:
    static void Save(IOutputStream* out, const NChromiumTrace::TInstantEvent& value);
    static void Load(IInputStream* in, NChromiumTrace::TInstantEvent& value, TMemoryPool& pool);
};

template <>
class TSerializer<NChromiumTrace::TAsyncEvent> {
public:
    static void Save(IOutputStream* out, const NChromiumTrace::TAsyncEvent& value);
    static void Load(IInputStream* in, NChromiumTrace::TAsyncEvent& value, TMemoryPool& pool);
};

template <>
class TSerializer<NChromiumTrace::TCounterEvent> {
public:
    static void Save(IOutputStream* out, const NChromiumTrace::TCounterEvent& value);
    static void Load(IInputStream* in, NChromiumTrace::TCounterEvent& value, TMemoryPool& pool);
};

template <>
class TSerializer<NChromiumTrace::TMetadataEvent> {
public:
    static void Save(IOutputStream* out, const NChromiumTrace::TMetadataEvent& value);
    static void Load(IInputStream* in, NChromiumTrace::TMetadataEvent& value, TMemoryPool& pool);
};

template <>
class TSerializer<NChromiumTrace::TEventWithArgs> {
public:
    static void Save(IOutputStream* out, const NChromiumTrace::TEventWithArgs& value);
    static void Load(IInputStream* in, NChromiumTrace::TEventWithArgs& value, TMemoryPool& pool);
};
