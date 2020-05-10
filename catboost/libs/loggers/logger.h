#pragma once

#include "tensorboard_logger.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>

#include <library/cpp/json/writer/json_value.h>
#include <util/stream/format.h>
#include <util/generic/hash.h>
#include <util/generic/ymath.h>


class IMetricEvalResult {
public:
    virtual double GetMetricValue() const = 0;
    virtual TString GetMetricName() const = 0;
    virtual TString BuildHumanReadableMetricString() const = 0;
    virtual bool IsMainMetric() const = 0;
    virtual ~IMetricEvalResult() = default;
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
    virtual void OutputParameters(const TString& /*sourceName*/, const NJson::TJsonValue& /*parameters*/) {}
    virtual void OutputProfile(const TProfileResults& /*profileResults*/) {}
    virtual void Flush(const int currentIteration) = 0;
};

inline bool DoOutputIteration(int currentIteration, int iterationsCount, int writePeriod) {
    return writePeriod > 0 && (currentIteration % writePeriod == 0 || currentIteration == iterationsCount - 1);
}

class TJsonLoggingBackend : public ILoggingBackend {
public:
    explicit TJsonLoggingBackend(const TString& fileName, const NJson::TJsonValue& metaJson, int writePeriod = 1)
        : File(fileName, CreateAlways)
        , WritePeriod(writePeriod)
        , IterationsCount(metaJson["iteration_count"].GetInteger())
    {
        TString metaString = "{\n\"meta\":" + ToString<NJson::TJsonValue>(metaJson) + ",\n\"iterations\":[\n]}";
        File.Write(metaString.data(), metaString.length());
    }

    void OutputMetric(const TString& sourceName, const IMetricEvalResult& evalResult) {
        double metricValue = evalResult.GetMetricValue();
        if (IsValidFloat(metricValue)) {
            IterationJson[sourceName].AppendValue(metricValue);
        } else {
            IterationJson[sourceName].AppendValue(ToString<double>(metricValue));
        }
    }

    void OutputParameters(const TString& sourceName, const NJson::TJsonValue& parameters) {
        IterationJson[sourceName].AppendValue(parameters);
    }

    void OutputProfile(const TProfileResults& profileResults) {
        IterationJson["remaining_time"] = profileResults.RemainingTime;
        IterationJson["passed_time"] = profileResults.PassedTime;
    }

    void Flush(const int currentIteration) {
        if (IterationJson.IsDefined() && DoOutputIteration(currentIteration, IterationsCount, WritePeriod)) {
            IterationJson.InsertValue("iteration", currentIteration);

            TString iterationInfo = ",";
            if (IsFirstIteration) {
                iterationInfo.clear();
                IsFirstIteration = false;
            }
            iterationInfo += "\n" + ToString<NJson::TJsonValue>(IterationJson) + "\n]}";

            File.Seek(-3, sCur);
            File.Write(iterationInfo.data(), iterationInfo.length());
        }
        IterationJson = NJson::JSON_UNDEFINED;
    }

private:
    bool IsFirstIteration = true;
    TFile File;
    int WritePeriod;
    int IterationsCount;
    NJson::TJsonValue IterationJson;
};

class TConsoleLoggingBackend : public ILoggingBackend {
public:
    explicit TConsoleLoggingBackend(const bool detailedProfile, int writePeriod = 1, int iterationsCount = 0)
        : DetailedProfile(detailedProfile)
        , WritePeriod(writePeriod)
        , IterationsCount(iterationsCount)
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
        if (!Stream.empty() && DoOutputIteration(currentIteration, IterationsCount, WritePeriod)) {
            CATBOOST_NOTICE_LOG << currentIteration << ":" << Stream.Str() << Endl;
        }
        Stream.Clear();
    }

private:
    bool DetailedProfile;
    int WritePeriod;
    int IterationsCount;
    TStringStream Stream;
};

class TProfileLoggingBackend : public ILoggingBackend {
public:
    explicit TProfileLoggingBackend(const TString& fileName)
        : File(new TOFStream(fileName))
    {
    }

    void OutputProfile(const TProfileResults& profileResults) {
        Stream << "\nProfile:" << Endl;
        for (const auto& it : profileResults.OperationToTime) {
            Stream << it.first << ": " << FloatToString(it.second, PREC_NDIGITS, 3) << " sec" << Endl;
        }
        Stream << "Passed: " << FloatToString(profileResults.CurrentTime, PREC_NDIGITS, 3) << " sec" << Endl;
        if (profileResults.IsIterationGood) {
            Stream << "\ttotal: " << HumanReadable(TDuration::Seconds(profileResults.PassedTime));
            Stream << "\tremaining: " << HumanReadable(TDuration::Seconds(profileResults.RemainingTime));
        }
        PassedIterations = profileResults.PassedIterations;
        OperationToTimeInAllIterations = profileResults.OperationToTimeInAllIterations;
    }

    void Flush(const int currentIteration) {
        *File << currentIteration << Stream.Str() << Endl;
        Stream.Clear();
    }

    ~TProfileLoggingBackend() {
        LogSummary();
    }

private:
    void LogSummary() {
        *File << Endl << "\nAverage times:" << Endl;
        if (PassedIterations == 0) {
            *File << Endl << "No iterations recorded" << Endl;
            return;
        }

        double time = OperationToTimeInAllIterations["Iteration time"] / PassedIterations;
        *File << "Iteration time: " << FloatToString(time, PREC_NDIGITS, 3) << " sec" << Endl;

        for (const auto& it : OperationToTimeInAllIterations) {
            *File << it.first << ": "
                << FloatToString(it.second / PassedIterations, PREC_NDIGITS, 3) << " sec" << Endl;
        }
    }

    THolder<TOFStream> File;
    TStringStream Stream;
    int PassedIterations;
    TMap<TString, double> OperationToTimeInAllIterations;
};

class TJsonProfileLoggingBackend : public ILoggingBackend {
public:
    explicit TJsonProfileLoggingBackend(const TString& fileName)
        : File(new TOFStream(fileName))
    {
    }

    void OutputProfile(const TProfileResults& profileResults) {
        CurrentValue = NJson::TJsonValue();
        CurrentValue["iteration"] = profileResults.PassedIterations;
        auto& times = CurrentValue["times"];
        for (const auto& it : profileResults.OperationToTime) {
            times[it.first] = it.second;
        }

        PassedIterations = profileResults.PassedIterations;
        OperationToTimeInAllIterations = profileResults.OperationToTimeInAllIterations;
    }

    void Flush(const int ) {
        *File << CurrentValue.GetStringRobust() << Endl;
    }

    ~TJsonProfileLoggingBackend() {
        LogSummary();
    }

private:
    void LogSummary() {
        if (PassedIterations == 0) {
            return;
        }
        CurrentValue = NJson::TJsonValue();
        CurrentValue["average_period"] = PassedIterations;

        double time = OperationToTimeInAllIterations["Iteration time"] / PassedIterations;
        CurrentValue["average_iteration_time"] = time;
        auto& times = CurrentValue["times"];
        for (const auto& it : OperationToTimeInAllIterations) {
            times[it.first] = it.second / PassedIterations;
        }
        *File << CurrentValue.GetStringRobust() << Endl;
    }
    NJson::TJsonValue CurrentValue;
    THolder<TOFStream> File;
    int PassedIterations;
    TMap<TString, double> OperationToTimeInAllIterations;
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
        if (!Stream.Empty()) {
            *File << currentIteration << Stream.Str() << Endl;
            Stream.Clear();
        }
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
    TLogger() = default;
    TLogger(int firstIteration, int lastIteration, int iterationBlockSize)
        : CurrentIteration(firstIteration)
        , LastIteration(lastIteration)
        , IterationBlockSize(iterationBlockSize)
    {
        Y_ASSERT(CurrentIteration >= 0);
        Y_ASSERT(LastIteration >= 0);
        Y_ASSERT(iterationBlockSize > 0);
    }

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

    void OutputParameters(const TString& sourceName, const NJson::TJsonValue& parameters) {
        for (auto& backend : Backends[sourceName]) {
            backend->OutputParameters(sourceName, parameters);
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
        CurrentIteration += IterationBlockSize;
        if (IterationBlockSize > 1) {
            CurrentIteration = Min(CurrentIteration, LastIteration);
        }
    }

    friend TOneInterationLogger;
    THashMap<TString, TVector<TIntrusivePtr<ILoggingBackend>>> Backends;
    TVector<TIntrusivePtr<ILoggingBackend>> ProfileOutputBackends;
    int CurrentIteration = 0;
    int LastIteration = 0;
    int IterationBlockSize = 1;
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

    void OutputParameters(const TString& sourceName, const NJson::TJsonValue& parameters) {
        Logger.OutputParameters(sourceName, parameters);
    }

    void OutputProfile(const TProfileResults& profileResults) {
        Logger.OutputProfile(profileResults);
    }
private:
    TLogger& Logger;
};

void LogAverages(const TProfileResults& profileResults);

class TImportanceLogger {
public:
    TImportanceLogger(
        size_t iterations,
        const TString& processedObjectToken,
        const TString& introductionLog,
        int writePeriod
    )
        : Iterations(iterations)
        , WritePeriod(writePeriod)
        , ProcessedObjectToken(processedObjectToken)
        , PassedIterations(0)
    {
        if (!writePeriod)
            return;
        CATBOOST_INFO_LOG << introductionLog << Endl;
    }

    void Log(const TProfileResults& profileResults) {
        if (!WritePeriod)
            return;
        size_t oldPassedIterations = PassedIterations;
        PassedIterations = profileResults.PassedIterations;

        if (GetLastWriteIterationsNumber() > oldPassedIterations) {
            CATBOOST_INFO_LOG << profileResults.PassedIterations << "/" << Iterations << " " << ProcessedObjectToken;
            CATBOOST_INFO_LOG << "\tpassed time: " << HumanReadable(TDuration::Seconds(profileResults.PassedTime));
            CATBOOST_INFO_LOG << "\tremaining time: " << HumanReadable(TDuration::Seconds(profileResults.RemainingTime));
            CATBOOST_INFO_LOG << Endl;
        }
    }
private:
    const size_t Iterations;
    const size_t WritePeriod;
    const TString ProcessedObjectToken;
    size_t PassedIterations;

    size_t GetLastWriteIterationsNumber() {
        if (PassedIterations == Iterations) {
            return Iterations - 1;
        }
        return ((PassedIterations - 1) / WritePeriod) * WritePeriod + 1;
    }
};
