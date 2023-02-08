#pragma once

#include <library/cpp/json/json_value.h>
#include <library/cpp/json/writer/json.h>
#include <library/cpp/threading/future/async.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/stream/format.h>
#include <util/string/builder.h>
#include <util/thread/pool.h>
#include <util/system/mutex.h>

namespace NSimpleHistogram {
    template <typename T>
    class THistogram {
    public:
        explicit THistogram(TVector<T>&& values)
            : Values_(std::move(values))
        {
        }

        size_t TotalCount() const {
            return Values_.size();
        }

        T ValueAtPercentile(double percentile) const {
            Y_ASSERT(!Values_.empty());
            Y_ASSERT(percentile >= 0.0 && percentile <= 1.0);

            const size_t index = static_cast<size_t>(percentile * Values_.size());
            return Values_[Min(Values_.size() - 1, index)];
        }

    private:
        TVector<T> Values_;
    };

    template <typename T>
    class THistogramCalcer {
    public:
        size_t TotalCount() const {
            return Values_.size();
        }

        void Reserve(size_t cnt) {
            Values_.reserve(cnt);
        }

        void RecordValue(T value) {
            Values_.push_back(value);
        }

        THistogram<T> Calc() {
            if (!IsSorted(Values_.begin(), Values_.end())) {
                Sort(Values_.begin(), Values_.end());
            }
            return THistogram<T>(std::move(Values_));
        }

    private:
        TVector<T> Values_;
    };

    template <typename T>
    class TMultiHistogramCalcer {
    public:
        void RecordValue(TStringBuf name, T value) {
            Calcers_[name].RecordValue(value);
        }

        THashMap<TString, THistogram<T>> Calc() {
            THashMap<TString, THistogram<T>> result;

            for (auto& calcer : Calcers_) {
                result.emplace(calcer.first, calcer.second.Calc());
            }

            return result;
        }

    private:
        THashMap<TString, THistogramCalcer<T>> Calcers_;
    };

    template <typename T>
    class TThreadSafeMultiHistogramCalcer {
    public:
        void RecordValue(TStringBuf name, T value) {
            TGuard<TMutex> guard(Mutex_);
            Calcer_.RecordValue(name, value);
        }

        THashMap<TString, THistogram<T>> Calc() {
            return Calcer_.Calc();
        }

    private:
        TMutex Mutex_;
        TMultiHistogramCalcer<T> Calcer_;
    };

    template <typename T>
    NJson::TJsonValue ToJson(const THistogram<T>& hist, const TVector<double>& percentiles) {
        NJson::TJsonValue json;

        for (double percentile : percentiles) {
            TStringBuilder name;
            name << "Q" << Prec(percentile * 100, PREC_POINT_DIGITS_STRIP_ZEROES, 2);
            json[name] = hist.ValueAtPercentile(percentile);
        }

        json["RecordCount"] = hist.TotalCount();

        return json;
    }

    template <typename T>
    NJson::TJsonValue ToJson(const THashMap<TString, THistogram<T>>& hists, const TVector<double>& percentiles) {
        NJson::TJsonValue json;

        for (const auto& p : hists) {
            json[p.first] = ToJson(p.second, percentiles);
        }

        return json;
    }

    template <typename T>
    TString ToJsonStr(const THashMap<TString, THistogram<T>>& hists, const TVector<double>& percentiles, bool format = true) {
        NJson::TJsonValue json = ToJson(hists, percentiles);

        NJsonWriter::TBuf buf;
        if (format) {
            buf.SetIndentSpaces(4);
        }

        return buf.WriteJsonValue(&json, true).Str();
    }

}
