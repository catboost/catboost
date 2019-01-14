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
    int PassedIterations = 0;
    int BadIterations = 0;
    double PassedTime = 0.0;
};

class TProfileInfo {
public:
    explicit TProfileInfo(int iterations = 0)
        : InitIterations(0)
        , IsIterationGood(true)
        , Iterations(iterations)
        , RemainingTime(0)
        , LocalPassedTime(0)
        , CurrentTime(0)
    {
    }

    const TProfileInfoData& DumpProfileInfo() const {
        return ProfileData;
    }

    void InitProfileInfo(TProfileInfoData&& profileData) {
        ProfileData = std::move(profileData);
        InitIterations = ProfileData.PassedIterations;
    }

    void StartIterationBlock() {
        CurrentTime = 0;
        Timer.Reset();
        OperationToTime.clear();
    }

    void StartNextIteration() {
        StartIterationBlock();
    }

    void AddOperation(const TString& operation) {
        double passedTime = Timer.PassedReset();
        CurrentTime += passedTime;
        OperationToTime[operation] += passedTime; // operations can be repeated in one iteration
    }

    void FinishIterationBlock(int blockSize) {
        CurrentTime += Timer.PassedReset();
        OperationToTime["Iteration time"] = CurrentTime;
        double averageTime = ProfileData.PassedIterations == InitIterations + ProfileData.BadIterations ?
                             std::numeric_limits<double>::max() :
                             ProfileData.PassedTime / (ProfileData.PassedIterations - InitIterations - ProfileData.BadIterations);
        ProfileData.PassedIterations += blockSize;
        if (CurrentTime < 0 || CurrentTime / blockSize / MAX_TIME_RATIO > averageTime) {
            CATBOOST_DEBUG_LOG << "\nIteration with suspicious time " << FloatToString(CurrentTime, PREC_NDIGITS, 3)
                << " sec ignored in overall statistics." << Endl;
            ProfileData.BadIterations += blockSize;
        } else {
            ProfileData.PassedTime += CurrentTime;
            LocalPassedTime += CurrentTime;
            for (const auto &it : OperationToTime) {
                ProfileData.OperationToTimeInAllIterations[it.first] += it.second;
            }
            RemainingTime = LocalPassedTime / (ProfileData.PassedIterations - InitIterations - ProfileData.BadIterations) * (Iterations - ProfileData.PassedIterations);
        }
        IsIterationGood = (ProfileData.PassedIterations != InitIterations + ProfileData.BadIterations);
    }

    void FinishIteration() {
        FinishIterationBlock(/*blockSize*/1);
    }

    TProfileResults GetProfileResults() const {
        return {
            ProfileData.PassedTime,
            RemainingTime,
            IsIterationGood,
            CurrentTime,
            ProfileData.PassedIterations,
            OperationToTime,
            ProfileData.OperationToTimeInAllIterations
        };
    }

private:
    static constexpr int MAX_TIME_RATIO = 100;
    TProfileInfoData ProfileData;
    TMap<TString, double> OperationToTime;
    THPTimer Timer;
    int InitIterations;
    bool IsIterationGood;
    const int Iterations;
    double RemainingTime;
    double LocalPassedTime;
    double CurrentTime;
};
