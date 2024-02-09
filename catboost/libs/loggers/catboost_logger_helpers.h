#pragma once

#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/libs/metrics/metric.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/maybe.h>

class TLogger;
struct TProfileResults;

struct TTimeInfo {
    explicit TTimeInfo(const TProfileResults& profileResults);
    TTimeInfo() = default;

    double IterationTime = 0;
    double PassedTime = 0;
    double RemainingTime = 0;
};

Y_DECLARE_PODTYPE(TTimeInfo);

struct TMetricsAndTimeLeftHistory {
    TVector<THashMap<TString, double> > LearnMetricsHistory;          // [iter][metric]
    TVector<TVector<THashMap<TString, double>>> TestMetricsHistory;   // [iter][test][metric]
    TVector<TTimeInfo> TimeHistory;                                   // [iter]

    TMaybe<size_t> BestIteration;  // For last test for eval metric or loss function
    THashMap<TString, double> LearnBestError;
    TVector<THashMap<TString, double>> TestBestError;

    Y_SAVELOAD_DEFINE(LearnMetricsHistory, TestMetricsHistory, TimeHistory, BestIteration, LearnBestError, TestBestError);

    // Serialization for model metadata without TimeHistory
    NJson::TJsonValue SaveMetrics() const;
    static TMetricsAndTimeLeftHistory LoadMetrics(const NJson::TJsonValue& rhs);

    void AddLearnError(const IMetric& metric, double error);
    void AddTestError(size_t testIdx, const IMetric& metric, double error, bool updateBestIteration);

private:
    void TryUpdateBestError(const IMetric& metric, double error, THashMap<TString, double>& bestError, bool updateBestIteration);
};


class TOutputFiles {
public:
    TOutputFiles(const NCatboostOptions::TOutputFilesOptions& params,
                 const TString& namesPrefix) {
        InitializeFiles(params, namesPrefix);
    }

    TString NamesPrefix;
    TString TimeLeftLogFile;
    TString LearnErrorLogFile;
    TString TestErrorLogFile;
    TString SnapshotFile;
    TString JsonLogFile;
    TString ProfileLogFile;
    TString TrainDir;
    TString ExperimentName;

    static TString AlignFilePath(const TString& baseDir, const TString& fileName, const TString& namePrefix = "");
    static TString AlignFilePathAndCreateDir(const TString& baseDir, const TString& fileName, const TString& namePrefix = "");

private:
    void InitializeFiles(const NCatboostOptions::TOutputFilesOptions& params, const TString& namesPrefix);
};


void AddFileLoggers(
    bool detailedProfile,
    const TString& learnErrorLogFile,
    const TString& testErrorLogFile,
    const TString& timeLogFile,
    const TString& jsonLogFile,
    const TString& profileLogFile,
    const TString& trainDir,
    const NJson::TJsonValue& metaJson,
    int metricPeriod,
    TLogger* logger
);

void AddConsoleLogger(
    const TString& learnToken,
    const TVector<const TString>& testTokens,
    bool hasTrain,
    int metricPeriod,
    int iterationsCount,
    TLogger* logger
);

void Log(
    int iteration,
    const TVector<TString>& metricsDescription,
    const TVector<THashMap<TString, double>>& learnErrorsHistory,
    const TVector<TVector<THashMap<TString, double>>>& testErrorsHistory, // [iter][test][metric]
    TMaybe<double> bestErrorValue,
    TMaybe<int> bestIteration,
    const TProfileResults& profileResults,
    const TString& learnToken,
    const TVector<const TString>& testTokens,
    bool outputMetrics,
    TLogger* logger
);



NJson::TJsonValue GetJsonMeta(
        int iterationCount,
        const TString& optionalExperimentName,
        const TVector<const IMetric*>& metrics,
        const TVector<TString>& learnSetNames,
        const TVector<TString>& testSetNames,
        const TString& parametersName,
        ELaunchMode launchMode
);

TString GetTrainModelLearnToken();
TVector<const TString> GetTrainModelTestTokens(int testCount);
TString GetParametersToken();


void InitializeFileLoggers(const NCatboostOptions::TCatBoostOptions& catboostOptions,
                           const TOutputFiles& outputFiles,
                           const TVector<const IMetric*>& metrics,
                           const TString& learnToken,
                           const TVector<const TString>& testTokens,
                           int metricPeriod,
                           TLogger* logger);


void InitializeFileLoggers(
    const NCatboostOptions::TOutputFilesOptions& outputFileOptions,
    const NJson::TJsonValue& metricsMetaJson,
    const TString& namesPrefix,
    bool isDetailedProfile,
    TLogger* logger);
