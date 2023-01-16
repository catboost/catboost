#pragma once

#include "event.h"
#include "sample_value.h"
#include "tracer.h"

namespace NChromiumTrace {
    class TTracer;

    class TCounter {
        TStringBuf Name;
        TStringBuf Categories;
        TEventArgs Args;

    public:
        TCounter(TStringBuf name)
            : TCounter(name, "sample"sv)
        {
        }

        TCounter(TStringBuf name, TStringBuf categories)
            : Name(name)
            , Categories(categories)
        {
        }

        template <typename T>
        TCounter& Sample(TStringBuf name, const T& sample) {
            if (auto value = GetValue(sample)) {
                Args.Add(name, *value);
            }
            return *this;
        }

        TCounter& Publish(TTracer& tracer) {
            if (Args.Items) {
                tracer.AddCounterNow(Name, Categories, Args);
            }
            return *this;
        }

    private:
        template <typename T>
        static TMaybe<T> GetValue(const TMaybe<T>& value) {
            return value;
        }

        template <typename T>
        static TMaybe<T> GetValue(const TSampleValue<T>& value) {
            return value.Value;
        }

        template <typename T>
        static TMaybe<T> GetValue(const TSampleValueWithTimestamp<T>& value) {
            return value.Value;
        }

        template <typename T>
        static TMaybe<double> GetValue(const TDerivativeSampleValue<T>& value) {
            return value.PositiveDerivative.Value;
        }
    };
}
