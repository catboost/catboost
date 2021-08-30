#include "pool.h"

#include <library/cpp/linear_regression/linear_regression.h>

#include <util/datetime/base.h>
#include <util/datetime/cputimer.h>

#include <util/system/type_name.h>

#include <util/string/printf.h>

template <typename TLRSolver>
void QualityBenchmark(const TPool& originalPool) {
    auto measure = [&](const double injureFactor, const double injureOffset) {
        TPool injuredPool = originalPool.InjurePool(injureFactor, injureOffset);

        static const size_t runsCount = 10;
        static const size_t foldsCount = 10;

        TMeanCalculator determinationCoefficientCalculator;

        TPool::TCVIterator learnIterator = injuredPool.CrossValidationIterator(foldsCount, TPool::LearnIterator);
        TPool::TCVIterator testIterator = injuredPool.CrossValidationIterator(foldsCount, TPool::TestIterator);

        for (size_t runNumber = 0; runNumber < runsCount; ++runNumber) {
            for (size_t foldNumber = 0; foldNumber < foldsCount; ++foldNumber) {
                learnIterator.ResetShuffle();
                learnIterator.SetTestFold(foldNumber);
                testIterator.ResetShuffle();
                testIterator.SetTestFold(foldNumber);

                TLRSolver solver;
                for (; learnIterator.IsValid(); ++learnIterator) {
                    solver.Add(learnIterator->Features, learnIterator->Goal, learnIterator->Weight);
                }
                TLinearModel model = solver.Solve();

                TDeviationCalculator goalsCalculator;
                TKahanAccumulator<double> errorsCalculator;
                for (; testIterator.IsValid(); ++testIterator) {
                    const double prediction = model.Prediction(testIterator->Features);
                    const double goal = testIterator->Goal;
                    const double weight = testIterator->Weight;
                    const double error = goal - prediction;

                    goalsCalculator.Add(goal, weight);
                    errorsCalculator += error * error * weight;
                }

                const double determinationCoefficient = 1 - errorsCalculator.Get() / goalsCalculator.GetDeviation();
                determinationCoefficientCalculator.Add(determinationCoefficient);
            }
        }

        return determinationCoefficientCalculator.GetMean();
    };

    Cout << TypeName<TLRSolver>() << ":\n";
    Cout << "\t" << Sprintf("base    : %.10lf\n", measure(1., 0.));
    Cout << "\t" << Sprintf("injure1 : %.10lf\n", measure(1e-1, 1e+1));
    Cout << "\t" << Sprintf("injure2 : %.10lf\n", measure(1e-3, 1e+4));
    Cout << "\t" << Sprintf("injure3 : %.10lf\n", measure(1e-3, 1e+5));
    Cout << "\t" << Sprintf("injure4 : %.10lf\n", measure(1e-3, 1e+6));
    Cout << "\t" << Sprintf("injure5 : %.10lf\n", measure(1e-4, 1e+6));
    Cout << "\t" << Sprintf("injure6 : %.10lf\n", measure(1e-4, 1e+7));
    Cout << Endl;
}

template <typename TLRSolver>
void SpeedBenchmark(const TPool& originalPool) {
    TDeviationCalculator speedTest;

    static const size_t runsCount = 1000;
    for (size_t runNumber = 0; runNumber < runsCount; ++runNumber) {
        TLRSolver solver;
        TLinearModel model;
        {
            TSimpleTimer timer;
            for (const TInstance& instance : originalPool) {
                solver.Add(instance.Features, instance.Goal, instance.Weight);
            }
            model = solver.Solve();

            speedTest.Add(timer.Get().MicroSeconds());
        }
    }

    const double multiplier = 1e-6;
    Cout << Sprintf("%.5lf +/- %.5lf: ", speedTest.GetMean() * multiplier, speedTest.GetStdDev() * multiplier) << TypeName<TLRSolver>() << Endl;
}

int main(int argc, const char** argv) {
    for (int taskNumber = 1; taskNumber < argc; ++taskNumber) {
        TPool pool;
        pool.ReadFromFeatures(argv[taskNumber]);

        Cout << argv[taskNumber] << ":" << Endl;
        QualityBenchmark<TFastBestSLRSolver>(pool);
        QualityBenchmark<TKahanBestSLRSolver>(pool);
        QualityBenchmark<TBestSLRSolver>(pool);

        QualityBenchmark<TLinearRegressionSolver>(pool);
        QualityBenchmark<TFastLinearRegressionSolver>(pool);

        SpeedBenchmark<TFastBestSLRSolver>(pool);
        SpeedBenchmark<TKahanBestSLRSolver>(pool);
        SpeedBenchmark<TBestSLRSolver>(pool);

        SpeedBenchmark<TLinearRegressionSolver>(pool);
        SpeedBenchmark<TFastLinearRegressionSolver>(pool);
    }

    return 0;
}
