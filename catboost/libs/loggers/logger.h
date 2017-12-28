#pragma once

#include "tensorboard_logger.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>

#include <util/stream/format.h>
#include <util/generic/hash.h>

class IMetricEvalResult {
public:
    virtual double GetMetricValue() const = 0;
    virtual TString GetMetricName() const = 0;
    virtual TString BuildHumanReadableMetricString() const = 0;
    virtual bool IsMainMetric() const = 0;
};

class TMetricEvalResult : public IMetricEvalResult {
public:
    TMetricEvalResult(const TString& name, const double value, const bool isMainMetric)
        : Name(name)
        , Value(value)
        , BestValue(0)
        , BestIteration(0)
        , IsMain(isMainMetric)
        , HaveBestResults(false)
    {
    }

    TMetricEvalResult(
        const TString& name,
        const double value,
        const double bestValue,
        const int bestIteration,
        const bool isMainMetric
    )
        : Name(name)
        , Value(value)
        , BestValue(bestValue)
        , BestIteration(bestIteration)
        , IsMain(isMainMetric)
        , HaveBestResults(true)
    {
    }

    double GetMetricValue() const override {
        return Value;
    }

    TString GetMetricName() const override {
        return Name;
    }

    bool IsMainMetric() const override {
        return IsMain;
    }

    TString BuildHumanReadableMetricString() const override {
        TStringStream result;
        result << Prec(Value, PREC_POINT_DIGITS,7);
        if (HaveBestResults) {
            result << "\tbest: " << Prec(BestValue, PREC_POINT_DIGITS, 7) << " (" << BestIteration << ")";
        }
        return result.Str();
    }

private:
    TString Name;
    double Value;
    double BestValue;
    int BestIteration;
    bool IsMain;
    bool HaveBestResults;
};

class ILoggingBackend : public TThrRefBase {
public:
    virtual void OutputMetric(const TString& /*sourceName*/, const IMetricEvalResult& /*evalResult*/) {}
    virtual void OutputProfile(const TProfileResults& /*profileResults*/) {}
    virtual void Flush(const int currentIteration) = 0;
};

class TConsoleLoggingBackend : public ILoggingBackend {
public:
    explicit TConsoleLoggingBackend(const bool detailedProfile, int writePeriod = 1)
        : DetailedProfile(detailedProfile)
        , WritePeriod(writePeriod)
    {
    }

    void OutputMetric(const TString& sourceName, const IMetricEvalResult& evalResult) {
        if (evalResult.IsMainMetric()) {
            Stream << "\t" << sourceName << ": " << evalResult.BuildHumanReadableMetricString();
        }
    }

    void OutputProfile(const TProfileResults& profileResults) {
        if (DetailedProfile) {
            Stream << "\nProfile:" << Endl;
            for (const auto& it : profileResults.OperationToTime) {
                Stream << it.first << ": " << FloatToString(it.second, PREC_NDIGITS, 3) << " sec" << Endl;
            }
            Stream << "Passed: " << FloatToString(profileResults.CurrentTime, PREC_NDIGITS, 3) << " sec" << Endl;
        }
        if (profileResults.IsIterationGood) {
            Stream << "\ttotal: " << HumanReadable(TDuration::Seconds(profileResults.PassedTime));
            Stream << "\tremaining: " << HumanReadable(TDuration::Seconds(profileResults.RemainingTime));
        }
    }

    void Flush(const int currentIteration) {
        if (currentIteration % WritePeriod != 0) {
            Stream.Clear();
            return;
        }
        if(!Stream.Empty()) {
            MATRIXNET_NOTICE_LOG << currentIteration << ":" << Stream.Str() << Endl;
            Stream.Clear();
        }
    }

private:
    bool DetailedProfile;
    int WritePeriod;
    TStringStream Stream;
};

class TErrorFileLoggingBackend : public ILoggingBackend {
public:
    explicit TErrorFileLoggingBackend(const TString& fileName)
        : File(new TOFStream(fileName))
    {
    }

    void OutputMetric(const TString& /*sourceName*/, const IMetricEvalResult& evalResult) {
        Stream << "\t" << evalResult.GetMetricValue();
        if (IsFirstIteration) {
            TitleStream << "\t" << evalResult.GetMetricName();
        }
    }

    void Flush(const int currentIteration) {
        if (IsFirstIteration) {
            *File << "iter" << TitleStream.Str() << Endl;
            IsFirstIteration = false;
        }
        *File << currentIteration << Stream.Str() << Endl;
        Stream.Clear();
    }

private:
    bool IsFirstIteration = true;
    TStringStream Stream;
    TStringStream TitleStream;
    THolder<TOFStream> File;
};

class TTimeFileLoggingBackend : public ILoggingBackend {
public:
    explicit TTimeFileLoggingBackend(const TString& fileName)
        : File(new TOFStream(fileName))
    {
    }

    void OutputProfile(const TProfileResults& profileResults) {
        Stream << "\t" << TDuration::Seconds(profileResults.PassedTime).MilliSeconds();
        Stream << "\t" << TDuration::Seconds(profileResults.RemainingTime).MilliSeconds();
        if (IsFirstIteration) {
            TitleStream << "\t" << "Passed\tRemaining";
        }
    }

    void Flush(const int currentIteration) {
        if (IsFirstIteration) {
            *File << "iter" << TitleStream.Str() << Endl;
            IsFirstIteration = false;
        }
        *File << currentIteration << Stream.Str() << Endl;
        Stream.Clear();
    }

private:
    bool IsFirstIteration = true;
    TStringStream Stream;
    TStringStream TitleStream;
    THolder<TOFStream> File;
};

class TTensorBoardLoggingBackend : public ILoggingBackend {
public:
    explicit TTensorBoardLoggingBackend(const TString& dirName)
        : Logger(new TTensorBoardLogger(dirName))
    {
    }

    void OutputMetric(const TString& /*sourceName*/, const IMetricEvalResult& evalResult) {
        MetricsInfo.emplace_back(evalResult.GetMetricName(), evalResult.GetMetricValue());
    }

    void Flush(const int currentIteration) {
        for (auto& metricInfo : MetricsInfo) {
            Logger->AddScalar(metricInfo.Name, currentIteration, metricInfo.Value);
        }
        MetricsInfo.clear();
    }

private:
    struct MetricInfo {
        MetricInfo(const TString& name, const double value)
            : Name(name), Value(value) {}
        TString Name;
        double Value;
    };

    TVector<MetricInfo> MetricsInfo;
    THolder<TTensorBoardLogger> Logger;
};

class TOneInterationLogger;

class TLogger {
public:
    void AddBackend(const TString& sourceName, TIntrusivePtr<ILoggingBackend> loggingBackend) {
        Backends[sourceName].push_back(loggingBackend);
    }

    void AddProfileBackend(TIntrusivePtr<ILoggingBackend> loggingBackend) {
        ProfileOutputBackends.push_back(loggingBackend);
    }

private:
    void OutputMetric(const TString& sourceName, const IMetricEvalResult& evalResult) {
        for (auto& backend : Backends[sourceName]) {
            backend->OutputMetric(sourceName, evalResult);
        }
    }

    void OutputProfile(const TProfileResults& profileResults) {
        for (auto& backend : ProfileOutputBackends) {
            backend->OutputProfile(profileResults);
        }
    }

    void FinishIteration() {
        for (auto& it : Backends) {
            for (auto backend : it.second) {
                backend->Flush(CurrentIteration);
            }
        }
        for (auto& backend : ProfileOutputBackends) {
            backend->Flush(CurrentIteration);
        }
        ++CurrentIteration;
    }

    friend TOneInterationLogger;
    THashMap<TString, TVector<TIntrusivePtr<ILoggingBackend>>> Backends;
    TVector<TIntrusivePtr<ILoggingBackend>> ProfileOutputBackends;
    int CurrentIteration = 0;
};

class TOneInterationLogger {
public:
    explicit TOneInterationLogger(TLogger& logger)
        : Logger(logger)
    {
    }

    ~TOneInterationLogger() {
        Logger.FinishIteration();
    }

    void OutputMetric(const TString& sourceName, const IMetricEvalResult& evalResult) {
        Logger.OutputMetric(sourceName, evalResult);
    }

    void OutputProfile(const TProfileResults& profileResults) {
        Logger.OutputProfile(profileResults);
    }
private:
    TLogger& Logger;
};

TLogger CreateLogger(
    const TVector<TString>& metricsDescription,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<double>>& testErrorsHistory,
    const TVector<TVector<double>>& timeHistory,
    const TString learnErrorLogFile,
    const TString testErrorLogFile,
    const TString timeLogFile,
    const TString trainDir,
    const bool allowWriteFiles,
    const bool detailedProfile,
    const bool hasTrain,
    const bool hasTest,
    int metricPeriod
);

void Log(
    const TVector<TString>& metricsDescription,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<double>>& testErrorsHistory,
    const double bestErrorValue,
    const int bestIteration,
    const TProfileResults& profileResults,
    TLogger* logger
);

void LogAverages(const TProfileResults& profileResults);
