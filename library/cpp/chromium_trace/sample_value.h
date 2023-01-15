#pragma once

#include <util/datetime/base.h>
#include <util/generic/maybe.h>

namespace NChromiumTrace {
    template <typename T>
    struct TSampleValue {
        TMaybe<T> Value;

        void Update(T value) {
            Value = value;
        }
    };

    template <typename T>
    struct TSampleValueWithTimestamp {
        TMaybe<T> Value;
        TMaybe<TInstant> LastUpdated;

        void Update(T value, TInstant last_updated = Now()) {
            Value = value;
            LastUpdated = last_updated;
        }
    };

    template <typename T>
    struct TDerivativeSampleValue {
        TSampleValueWithTimestamp<T> Value;
        TSampleValue<double> PositiveDerivative;
        TSampleValue<double> NegativeDerivative;

        void Update(T sample, TInstant now = Now()) {
            if (Value.Value) {
                auto dt = Max<i64>(1, (now - *Value.LastUpdated).MicroSeconds());
                auto dx = sample - *Value.Value;
                auto derivative = double(dx) / dt;

                PositiveDerivative.Value = Max(0.0, derivative);
                NegativeDerivative.Value = Max(0.0, -derivative);
            }
            Value.Update(sample, now);
        }
    };
}
