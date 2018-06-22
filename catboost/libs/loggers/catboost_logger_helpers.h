#pragma once

#include "logger.h"
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/metrics/metric.h>

struct TTimeInfo {
    TTimeInfo(double passedTime, double remainingTime)
        : PassedTime(passedTime)
        , RemainingTime(remainingTime)
    {
    }
    TTimeInfo() = default;
    double PassedTime = 0;
    double RemainingTime = 0;
};

Y_DECLARE_PODTYPE(TTimeInfo);

struct TMetricsAndTimeLeftHistory {
    TVector<TVector<double> > LearnMetricsHistory;
    TVector<TVector<TVector<double>>> TestMetricsHistory;
    TVector<TTimeInfo> TimeHistory;


    Y_SAVELOAD_DEFINE(LearnMetricsHistory, TestMetricsHistory, TimeHistory);
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
    TString MetaFile;
    TString JsonLogFile;
    TString ProfileLogFile;
    TString TrainDir;
    TString ExperimentName;

    static TString AlignFilePath(const TString& baseDir, const TString& fileName, const TString& namePrefix = "");
    static TString AlignFilePathAndCreateDir(const TString& baseDir, const TString& fileName, const TString& namePrefix = "");

private:
    void InitializeFiles(const NCatboostOptions::TOutputFilesOptions& params, const TString& namesPrefix);
};


void WriteHistory(
    const TVector<TString>& metricsDescription,
    const TMetricsAndTimeLeftHistory& history,
    const TString& learnToken,
    const TVector<const TString>& testTokens,
    TLogger* logger
);

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
    const TVector<TString>& metricsDescription,
    const TVector<bool>& skipMetricOnTrain,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<TVector<double>>>& testErrorsHistory, // [iter][test][metric]
    double bestErrorValue,
    int bestIteration,
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
        ELaunchMode launchMode
);


void CreateMetaFile(const TOutputFiles& outputFiles,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    const TVector<const IMetric*>& losses,
                    ui32 iterationsCount);

TString GetTrainModelLearnToken();
TVector<const TString> GetTrainModelTestTokens(int testCount);


void InitializeFileLoggers(const NCatboostOptions::TCatBoostOptions& catboostOptions,
                           const TOutputFiles& outputFiles,
                           const TVector<const IMetric*>& metrics,
                           const TString& learnToken,
                           const TVector<const TString>& testTokens,
                           int metricPeriod,
                           TLogger* logger);

