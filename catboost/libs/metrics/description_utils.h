#pragma once
#include "metric.h"

#include <util/generic/string.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/string/printf.h>

#include <catboost/private/libs/options/enums.h>

template <typename T>
static inline TString BuildDescription(const TMetricParam<T>& param) {
    if (param.IsUserDefined()) {
        return TStringBuilder() << param.GetName() << "=" << ToString(param.Get());
    }
    return {};
}

template <>
inline TString BuildDescription<bool>(const TMetricParam<bool>& param) {
    if (param.IsUserDefined()) {
        return TStringBuilder() << param.GetName() << "=" << (param.Get() ? "true" : "false");
    }
    return {};
}

template <typename T>
static inline TString BuildDescription(const char* fmt, const TMetricParam<T>& param) {
    if (param.IsUserDefined()) {
        return TStringBuilder() << param.GetName() << "=" << Sprintf(fmt, param.Get());
    }
    return {};
}

template <>
inline TString BuildDescription(const char* fmt, const TMetricParam<TVector<double>>& param) {
    if (param.IsUserDefined() && param.Get().size() > 0) {
        TStringBuilder description;
        description << param.GetName() << "=" << Sprintf(fmt, param.Get()[0]);
        for (auto idx : xrange<size_t>(1, param.Get().size(), 1)) {
            description << "," << Sprintf(fmt, param.Get()[idx]);
        }
        return description;
    }
    return {};
}

template <typename T, typename... TRest>
static inline TString BuildDescription(const TMetricParam<T>& param, const TRest&... rest) {
    const TString& head = BuildDescription(param);
    const TString& tail = BuildDescription(rest...);
    const TString& sep = (head.empty() || tail.empty()) ? "" : ";";
    return TStringBuilder() << head << sep << tail;
}

template <typename T, typename... TRest>
static inline TString BuildDescription(const char* fmt, const TMetricParam<T>& param, const TRest&... rest) {
    const TString& head = BuildDescription(fmt, param);
    const TString& tail = BuildDescription(rest...);
    const TString& sep = (head.empty() || tail.empty()) ? "" : ";";
    return TStringBuilder() << head << sep << tail;
}

template <typename... TParams>
static inline TString BuildDescription(ELossFunction lossFunction, const TParams&... params) {
    const TString& tail = BuildDescription(params...);
    const TString& sep = tail.empty() ? "" : ":";
    return TStringBuilder() << ToString(lossFunction) << sep << tail;
}

template <typename... TParams>
static inline TString BuildDescription(const TString& description, const TParams&... params) {
    Y_ASSERT(!description.empty());
    const TString& tail = BuildDescription(params...);
    const TString& sep = tail.empty() ? "" : description.Contains(':') ? ";" : ":";
    return TStringBuilder() << description << sep << tail;
}

TString BuildDescriptionFromParams(ELossFunction lossFunction, const TLossParams& params);

static inline TMetricParam<double> MakeTargetBorderParam(double targetBorder) {
    return {"border", targetBorder, targetBorder != GetDefaultTargetBorder()};
}

static inline TMetricParam<double> MakePredictionBorderParam(double predictionBorder) {
    return {NCatboostOptions::TMetricOptions::PREDICTION_BORDER_PARAM, predictionBorder, predictionBorder != GetDefaultPredictionBorder()};
}
