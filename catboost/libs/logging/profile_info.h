#pragma once

#include <catboost/libs/logging/logging.h>

#include <util/generic/map.h>
#include <util/stream/file.h>
#include <util/stream/format.h>
#include <util/system/hp_timer.h>

class TProfileInfo {
public:
    TProfileInfo(bool detailedProfile, int iterations, TOFStream* timeLeftLog)
        : PassedIterations(0)
        , InitIterations(0)
        , BadIterations(0)
        , DetailedProfile(detailedProfile)
        , Iterations(iterations)
        , PassedTime(0)
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
        , CurrentTime(0)
        , TimeLeftLog(nullptr) {
    }

    void SetInitIterations(int iter) {
        PassedIterations = iter;
        InitIterations = iter;
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
            MATRIXNET_INFO_LOG << "\nIteration with suspicious time " << FloatToString(CurrentTime, PREC_NDIGITS, 3) << " sec ignored in overall statistics." << Endl;
            BadIterations++;
        } else {
            PassedTime += CurrentTime;
            for (const auto &it : OperationToTime) {
                OperationToTimeInAllIterations[it.first] += it.second;
            }
            PrintState();
        }
    }

    void PrintState() const {
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
            remainingTime = PassedTime / (PassedIterations - InitIterations - BadIterations) * (Iterations - PassedIterations);
            log << "\ttotal: " << HumanReadable(TDuration::Seconds(PassedTime));
            log << "\tremaining: " << HumanReadable(TDuration::Seconds(remainingTime));
        }
        MATRIXNET_INFO_LOG << log.Str() << Endl;
        if (TimeLeftLog) {
            *TimeLeftLog << PassedIterations - 1 << "\t" << TDuration::Seconds(remainingTime).MilliSeconds() << "\t"
                         << TDuration::Seconds(PassedTime).MilliSeconds() << Endl;
        }
    }

    void PrintAverages() const {
        MATRIXNET_INFO_LOG << Endl << "Average times:" << Endl;

        double time = 0;
        for (const auto& it : OperationToTimeInAllIterations) {
            time += it.second;
        }
        time /= PassedIterations;
        MATRIXNET_INFO_LOG << "Iteration time: " << FloatToString(time, PREC_NDIGITS, 3) << " sec" << Endl;

        for (const auto& it : OperationToTimeInAllIterations) {
            MATRIXNET_INFO_LOG << it.first << ": " << FloatToString(it.second / PassedIterations, PREC_NDIGITS, 3) << " sec" << Endl;
        }
        MATRIXNET_INFO_LOG << Endl;
    }

private:
    static constexpr int MAX_TIME_RATIO = 100;
    ymap<TString, double> OperationToTime;
    ymap<TString, double> OperationToTimeInAllIterations;
    THPTimer Timer;
    int PassedIterations;
    int InitIterations;
    int BadIterations;
    bool DetailedProfile;
    const int Iterations;
    double PassedTime;
    double CurrentTime;
    TOFStream* TimeLeftLog;
};
