#pragma once

#include "logging.h"

#include <util/ysaveload.h>
#include <util/generic/map.h>
#include <util/stream/file.h>
#include <util/stream/format.h>
#include <util/system/hp_timer.h>

struct TProfileResults {
    TProfileResults(
        double passedTime,
        double remainingTime,
        bool isIterationGood = true,
        double currentTime = 0,
        int passedIterations = 0,
        TMap<TString, double> operationToTime = {},
        TMap<TString, double> operationToTimeInAllIterations = {}
    )
        : PassedTime(passedTime)
        , RemainingTime(remainingTime)
        , IsIterationGood(isIterationGood)
        , CurrentTime(currentTime)
        , PassedIterations(passedIterations)
        , OperationToTime(operationToTime)
        , OperationToTimeInAllIterations(operationToTimeInAllIterations)
    {
    }

    double PassedTime;
    double RemainingTime;
    bool IsIterationGood;
    double CurrentTime;
    int PassedIterations;
    TMap<TString, double> OperationToTime;
    TMap<TString, double> OperationToTimeInAllIterations;
};

struct TProfileInfoData {
    TProfileInfoData() = default;
    TProfileInfoData(
        const TMap<TString, double>& operationToTimeInAllIterations,
        int passedIterations,
        int badIterations,
        double passedTime
    )
        : OperationToTimeInAllIterations(operationToTimeInAllIterations)
        , PassedIterations(passedIterations)
        , BadIterations(badIterations)
        , PassedTime(passedTime)
    {
    }

    void Save(IOutputStream* s) const {
        ::SaveMany(s, OperationToTimeInAllIterations, PassedIterations, BadIterations, PassedTime);
    }
    void Load(IInputStream* s) {
        ::LoadMany(s, OperationToTimeInAllIterations, PassedIterations, BadIterations, PassedTime);
    }

    TMap<TString, double> OperationToTimeInAllIterations;
    int PassedIterations;
    int BadIterations;
    double PassedTime;
};

class TProfileInfo {
public:
    explicit TProfileInfo(int iterations = 0)
        : PassedIterations(0)
        , InitIterations(0)
        , BadIterations(0)
        , IsIterationGood(true)
        , Iterations(iterations)
        , PassedTime(0)
        , RemainingTime(0)
        , LocalPassedTime(0)
        , CurrentTime(0)
    {
    }

    TProfileInfoData DumpProfileInfo() const {
        return {OperationToTimeInAllIterations, PassedIterations, BadIterations, PassedTime};
    }

    void InitProfileInfo(TProfileInfoData&& profileData) {
        PassedIterations = profileData.PassedIterations;
        InitIterations = PassedIterations;
        BadIterations = profileData.BadIterations;
        PassedTime = profileData.PassedTime;
        OperationToTimeInAllIterations = std::move(profileData.OperationToTimeInAllIterations);
    }

    void StartNextIteration() {
        CurrentTime = 0;
        Timer.Reset();
        OperationToTime.clear();
    }

    void AddOperation(const TString& operation) {
        double passedTime = Timer.PassedReset();
        CurrentTime += passedTime;
        OperationToTime[operation] += passedTime; // operations can be repeated in one iteration
    }

    void FinishIteration() {
        CurrentTime += Timer.PassedReset();
        double averageTime = PassedIterations == InitIterations + BadIterations ?
                             std::numeric_limits<double>::max() :
                             PassedTime / (PassedIterations - InitIterations - BadIterations);
        ++PassedIterations;
        if (CurrentTime < 0 || CurrentTime / MAX_TIME_RATIO > averageTime) {
            MATRIXNET_WARNING_LOG << "\nIteration with suspicious time " << FloatToString(CurrentTime, PREC_NDIGITS, 3)
                << " sec ignored in overall statistics." << Endl;
            ++BadIterations;
        } else {
            PassedTime += CurrentTime;
            LocalPassedTime += CurrentTime;
            for (const auto &it : OperationToTime) {
                OperationToTimeInAllIterations[it.first] += it.second;
            }
            RemainingTime = LocalPassedTime / (PassedIterations - InitIterations - BadIterations) * (Iterations - PassedIterations);
        }
        IsIterationGood = (PassedIterations != InitIterations + BadIterations);
    }

    TProfileResults GetProfileResults() {
        return {
            PassedTime,
            RemainingTime,
            IsIterationGood,
            CurrentTime,
            PassedIterations,
            OperationToTime,
            OperationToTimeInAllIterations
        };
    }



private:
    static constexpr int MAX_TIME_RATIO = 100;
    TMap<TString, double> OperationToTime;
    TMap<TString, double> OperationToTimeInAllIterations;
    THPTimer Timer;
    int PassedIterations;
    int InitIterations;
    int BadIterations;
    bool IsIterationGood;
    const int Iterations;
    double PassedTime;
    double RemainingTime;
    double LocalPassedTime;
    double CurrentTime;
};
