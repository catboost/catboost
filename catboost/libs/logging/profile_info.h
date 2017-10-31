#pragma once

#include <catboost/libs/logging/logging.h>

#include <util/ysaveload.h>
#include <util/generic/map.h>
#include <util/stream/file.h>
#include <util/stream/format.h>
#include <util/system/hp_timer.h>

struct TProfileInfoData {
    TProfileInfoData() = default;
    TProfileInfoData(const ymap<TString, double>& operationToTimeInAllIterations,
            const yvector<yvector<ui64>>& timeLeftHistory, int passedIterations,
            int badIterations, double passedTime)
        : OperationToTimeInAllIterations(operationToTimeInAllIterations)
        , TimeLeftHistory(timeLeftHistory)
        , PassedIterations(passedIterations)
        , BadIterations(badIterations)
        , PassedTime(passedTime)
        {
        }

    void Save(IOutputStream* s) const {
        ::SaveMany(s, OperationToTimeInAllIterations, TimeLeftHistory, PassedIterations, BadIterations, PassedTime);
    }
    void Load(IInputStream* s) {
        ::LoadMany(s, OperationToTimeInAllIterations, TimeLeftHistory, PassedIterations, BadIterations, PassedTime);
    }

    ymap<TString, double> OperationToTimeInAllIterations;
    yvector<yvector<ui64>> TimeLeftHistory;
    int PassedIterations;
    int BadIterations;
    double PassedTime;
};

class TProfileInfo {
public:
    TProfileInfo(bool detailedProfile, int iterations, TOFStream* timeLeftLog)
        : PassedIterations(0)
        , InitIterations(0)
        , BadIterations(0)
        , DetailedProfile(detailedProfile)
        , Iterations(iterations)
        , PassedTime(0)
        , LocalPassedTime(0)
        , CurrentTime(0)
        , TimeLeftLog(timeLeftLog) {
    }

    explicit TProfileInfo(bool detailedProfile)
        : PassedIterations(0)
        , InitIterations(0)
        , BadIterations(0)
        , DetailedProfile(detailedProfile)
        , Iterations(0)
        , PassedTime(0)
        , LocalPassedTime(0)
        , CurrentTime(0)
        , TimeLeftLog(nullptr) {
    }

    TProfileInfoData DumpProfileInfo() const {
        return {OperationToTimeInAllIterations, TimeLeftHistory, PassedIterations, BadIterations, PassedTime};
    }

    void InitProfileInfo(TProfileInfoData&& profileData) {
        PassedIterations = profileData.PassedIterations;
        InitIterations = PassedIterations;
        BadIterations = profileData.BadIterations;
        PassedTime = profileData.PassedTime;
        TimeLeftHistory = std::move(profileData.TimeLeftHistory);
        OperationToTimeInAllIterations = std::move(profileData.OperationToTimeInAllIterations);
        for (int iteration = 0; iteration < TimeLeftHistory.ysize(); ++iteration) {
            *TimeLeftLog << iteration << "\t" << TimeLeftHistory[iteration][0]
                         << "\t" << TimeLeftHistory[iteration][1] << Endl;
        }
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
        PassedIterations++;
        if (CurrentTime < 0 || CurrentTime / MAX_TIME_RATIO > averageTime) {
            MATRIXNET_WARNING_LOG << "\nIteration with suspicious time " << FloatToString(CurrentTime, PREC_NDIGITS, 3) << " sec ignored in overall statistics." << Endl;
            BadIterations++;
        } else {
            PassedTime += CurrentTime;
            LocalPassedTime += CurrentTime;
            for (const auto &it : OperationToTime) {
                OperationToTimeInAllIterations[it.first] += it.second;
            }
            LogState();
        }
    }

    void LogState() {
        TStringStream log;
        if (DetailedProfile) {
            log << "\nProfile:" << Endl;
            for (const auto& it : OperationToTime) {
                log << it.first << ": " << FloatToString(it.second, PREC_NDIGITS, 3) << " sec" << Endl;
            }
            log << "Passed: " << FloatToString(CurrentTime, PREC_NDIGITS, 3) << " sec" << Endl;
        }
        double remainingTime = 0;
        if (PassedIterations != InitIterations + BadIterations) {
            remainingTime = LocalPassedTime / (PassedIterations - InitIterations - BadIterations) * (Iterations - PassedIterations);
            log << "\ttotal: " << HumanReadable(TDuration::Seconds(PassedTime));
            log << "\tremaining: " << HumanReadable(TDuration::Seconds(remainingTime));
        }
        MATRIXNET_NOTICE_LOG << log.Str() << Endl;
        if (TimeLeftLog) {
            TimeLeftHistory.push_back({TDuration::Seconds(remainingTime).MilliSeconds(),
                                       TDuration::Seconds(PassedTime).MilliSeconds()});
            *TimeLeftLog << PassedIterations - 1 << "\t" << TimeLeftHistory.back()[0]
                         << "\t" << TimeLeftHistory.back()[1] << Endl;
        }
    }

    void PrintAverages() const {
        MATRIXNET_NOTICE_LOG << Endl << "Average times:" << Endl;
        if (PassedIterations == 0) {
            MATRIXNET_NOTICE_LOG << Endl << "No iterations recorded" << Endl;
            return;
        }

        double time = 0;
        for (const auto& it : OperationToTimeInAllIterations) {
            time += it.second;
        }
        time /= PassedIterations;
        MATRIXNET_NOTICE_LOG << "Iteration time: " << FloatToString(time, PREC_NDIGITS, 3) << " sec" << Endl;

        for (const auto& it : OperationToTimeInAllIterations) {
            MATRIXNET_NOTICE_LOG << it.first << ": " << FloatToString(it.second / PassedIterations, PREC_NDIGITS, 3) << " sec" << Endl;
        }
        MATRIXNET_NOTICE_LOG << Endl;
    }

private:
    static constexpr int MAX_TIME_RATIO = 100;
    ymap<TString, double> OperationToTime;
    ymap<TString, double> OperationToTimeInAllIterations;
    yvector<yvector<ui64>> TimeLeftHistory;
    THPTimer Timer;
    int PassedIterations;
    int InitIterations;
    int BadIterations;
    bool DetailedProfile;
    const int Iterations;
    double PassedTime;
    double LocalPassedTime;
    double CurrentTime;
    TOFStream* TimeLeftLog;
};
