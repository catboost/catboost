#include "roc_curve.h"

#include "apply.h"

#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/target/data_providers.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/stream/fwd.h>
#include <util/string/cast.h>

#include <cmath>


using namespace NCB;


namespace {
    struct TClassWithProbability {
        int ClassId = 0;
        double Probability = 0.0;

    public:
        TClassWithProbability() = default;
        TClassWithProbability(int classId, double probability)
            : ClassId(classId)
            , Probability(probability)
        {}
    };
}

TRocPoint TRocCurve::IntersectSegments(const TRocPoint& leftEnds, const TRocPoint& rightEnds) {
    double x1 = leftEnds.Boundary, x2 = rightEnds.Boundary;
    double y11 = leftEnds.FalseNegativeRate, y21 = rightEnds.FalseNegativeRate;
    double y12 = leftEnds.FalsePositiveRate, y22 = rightEnds.FalsePositiveRate;
    double x = x1 + (x1 - x2) * (y11 - y12) / ((y21 - y22) - (y11 - y12));
    double y;
    if ((y22 - y12) < EPS) {
        y = 0.5 * (y12 + y22);
    } else if ((y11 - y21) < EPS) {
        y = 0.5 * (y11 + y21);
    } else {
        y = y11 + (x1 - x) * (y21 - y11) / (x1 - x2);
    }
    return TRocPoint(x, y, y);
}

void TRocCurve::AddPoint(double newBoundary, double newFnr, double newFpr) {
    TRocPoint newPoint(newBoundary, newFnr, newFpr);

    if (Points.size()) {
        double oldFnr = Points.back().FalseNegativeRate;
        double oldFpr = Points.back().FalsePositiveRate;
        if (oldFpr < oldFnr && newFpr > newFnr) {
            // will happen at least once: first point (1, 1, 0) satisfies first inequality,
            // last point (0, 0, 1) satisfies second inequality
            // if at some point equality stands, it is the intersection point itself
            RateCurvesIntersection = Points.size();
            Points.push_back(IntersectSegments(Points.back(), newPoint));
        }
    }

    Points.push_back(newPoint);
}

void TRocCurve::BuildCurve(
    const TVector<TVector<double>>& approxes, // [poolId][docId]
    const TVector<TConstArrayRef<float>>& labels, // [poolId][docId]
    NPar::ILocalExecutor* localExecutor
) {
    size_t allDocumentsCount = 0;
    for (const auto& label : labels) {
        allDocumentsCount += label.size();
    }
    TVector<TClassWithProbability> probabilitiesWithTargets(allDocumentsCount);

    TVector<size_t> countTargets(2, 0);

    size_t allDocumentsOffset = 0;
    for (size_t poolIdx = 0; poolIdx < labels.size(); ++poolIdx) {
        TVector<TVector<double>> rawApproxesMulti(1, approxes[poolIdx]);
        auto probabilities = PrepareEval(
            EPredictionType::Probability,
            /* ensemblesCount */ 1,
            /* lossFunctionName */ "",
            rawApproxesMulti,
            localExecutor);
        const auto& targets = labels[poolIdx];
        size_t documentsCount = targets.size();
        for (size_t documentIdx = 0; documentIdx < documentsCount; ++documentIdx) {
            int target = targets[documentIdx] + 0.5 /* custom round for accuracy */;
            ++countTargets[target];
            probabilitiesWithTargets[allDocumentsOffset + documentIdx] = TClassWithProbability(
                target,
                probabilities[0][documentIdx]
            );
        }
        allDocumentsOffset += documentsCount;
    };

    for (int classId : {0, 1}) {
        CB_ENSURE(
            countTargets[classId] > 0,
            "No documents of class " << ToString(classId) << "."
        );
    }

    StableSort(
        probabilitiesWithTargets.begin(),
        probabilitiesWithTargets.end(),
        [](const TClassWithProbability& element1, const TClassWithProbability& element2) {
            return element1.Probability > element2.Probability;
        }
    );

    Points.clear();
    Points.reserve(allDocumentsCount + 1);

    TVector<size_t> countTargetsIntermediate(2, 0);

    AddPoint(1, 1, 0); // always starts with (1, 1, 0)
    for (size_t pointIdx = 0; pointIdx < allDocumentsCount - 1; ++pointIdx) {
        ++countTargetsIntermediate[probabilitiesWithTargets[pointIdx].ClassId];

        double boundary
            = 0.5 * (probabilitiesWithTargets[pointIdx].Probability
                + probabilitiesWithTargets[pointIdx + 1].Probability);

        if (probabilitiesWithTargets[pointIdx + 1].Probability
            < (probabilitiesWithTargets[pointIdx].Probability - EPS))
        {
            double newFnr = double(countTargets[1] - countTargetsIntermediate[1]) / countTargets[1];
            double newFpr = double(countTargetsIntermediate[0]) / countTargets[0];

            AddPoint(boundary, newFnr, newFpr);
        }
    }
    AddPoint(0, 0, 1); // always ends with (0, 0, 1)
}


TRocCurve::TRocCurve(
    const TVector<TVector<double>>& approxes,
    const TVector<TConstArrayRef<float>>& labels,
    int threadCount
) {
    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    BuildCurve(approxes, labels, &localExecutor);
}


TRocCurve::TRocCurve(const TFullModel& model, const TVector<TDataProviderPtr>& datasets, int threadCount) {
    TVector<TVector<double>> approxes(datasets.size());
    TVector<TConstArrayRef<float>> labels(datasets.size());

    // need to save owners of labels data
    TVector<TTargetDataProviderPtr> targetDataParts(datasets.size());

    NCatboostOptions::TLossDescription logLoss;
    logLoss.LossFunction.Set(ELossFunction::Logloss);

    TRestorableFastRng64 rand(0);

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    localExecutor.ExecRange(
        [&] (int i) {
            TProcessedDataProvider processedData = CreateModelCompatibleProcessedDataProvider(
                *datasets[i],
                TConstArrayRef<NCatboostOptions::TLossDescription>(&logLoss, 1),
                model,
                GetMonopolisticFreeCpuRam(),
                &rand,
                &localExecutor
            );

            approxes[i] = ApplyModelMulti(
                model,
                *processedData.ObjectsData,
                EPredictionType::RawFormulaVal,
                0,
                0,
                &localExecutor,
                processedData.TargetData->GetBaseline()
            )[0];

            targetDataParts[i] = std::move(processedData.TargetData);

            labels[i] = *targetDataParts[i]->GetOneDimensionalTarget();
        },
        0,
        SafeIntegerCast<int>(datasets.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );

    BuildCurve(approxes, labels, &localExecutor);
}

static void CheckRocPoint(const TRocPoint& rocPoint) {
    CB_ENSURE(
        0.0 <= rocPoint.Boundary && rocPoint.Boundary <= 1.0,
        "Invalid boundary. Must be in [0.0, 1.0]."
    );
    CB_ENSURE(
        0.0 <= rocPoint.FalseNegativeRate && rocPoint.FalseNegativeRate <= 1.0,
        "Invalid FNR. Must be in [0.0, 1.0]."
    );
    CB_ENSURE(
        0.0 <= rocPoint.FalsePositiveRate && rocPoint.FalsePositiveRate <= 1.0,
        "Invalid FPR. Must be in [0.0, 1.0]."
    );
}

TRocCurve::TRocCurve(const TVector<TRocPoint>& points) {
    CB_ENSURE(points.size() >= 2, "ROC curve must have at least two points.");
    CB_ENSURE(
        points[0].Boundary == 1.0 &&
        points[0].FalseNegativeRate == 1.0 &&
        points[0].FalsePositiveRate == 0.0,
        "ROC curve must start with (1.0, 1.0, 0.0) point."
    );
    CB_ENSURE(
        points.back().Boundary == 0.0 &&
        points.back().FalseNegativeRate == 0.0 &&
        points.back().FalsePositiveRate == 1.0,
        "ROC curve must end with (0.0, 0.0, 1.0) point."
    );
    bool intersectionFound = false;
    for (size_t pointIdx = 1; pointIdx < points.size(); ++pointIdx) {
        CheckRocPoint(points[pointIdx]);
        CB_ENSURE(
            points[pointIdx - 1].Boundary > points[pointIdx].Boundary &&
            points[pointIdx - 1].FalseNegativeRate >= points[pointIdx].FalseNegativeRate &&
            points[pointIdx - 1].FalsePositiveRate <= points[pointIdx].FalsePositiveRate &&
            (
                points[pointIdx - 1].FalseNegativeRate > points[pointIdx].FalseNegativeRate ||
                points[pointIdx - 1].FalsePositiveRate < points[pointIdx].FalsePositiveRate
            ),
            "ROC curve points must be strictly ordered."
        );
        if (fabs(points[pointIdx].FalseNegativeRate - points[pointIdx].FalsePositiveRate) < 1e-12) {
            intersectionFound = true;
            RateCurvesIntersection = pointIdx;
        }
    }
    CB_ENSURE(intersectionFound, "FNR and FPR curves must intersect in some point.");
    Points = points;
}

double TRocCurve::SelectDecisionBoundaryByFalsePositiveRate(double falsePositiveRate) {
    CB_ENSURE(Points.size() > 0, "ROC curve must be non-empty.");
    CB_ENSURE(
        falsePositiveRate >= 0.0 && falsePositiveRate <= 1.0,
        "Invalid FPR value: " << ToString(falsePositiveRate) << ". Must be in [0.0, 1.0]."
    );

    auto cutElement = UpperBound(
        Points.begin(),
        Points.end(),
        falsePositiveRate,
        [](double falsePositiveRate, const TRocPoint& point) {
            return falsePositiveRate < point.FalsePositiveRate;
        }
    );
    --cutElement; // begin() has FPR == 0, so it will not be cutElement

    return cutElement->Boundary;
}

double TRocCurve::SelectDecisionBoundaryByFalseNegativeRate(double falseNegativeRate) {
    CB_ENSURE(Points.size() > 0, "ROC curve must be non-empty.");
    CB_ENSURE(
        falseNegativeRate >= 0.0 && falseNegativeRate <= 1.0,
        "Invalid FNR value: " << ToString(falseNegativeRate) << ". Must be in [0.0, 1.0]."
    );

    auto cutElement = UpperBound(
        Points.rbegin(),
        Points.rend(),
        falseNegativeRate,
        [](double falseNegativeRate, const TRocPoint& point) {
            return falseNegativeRate < point.FalseNegativeRate;
        }
    );
    --cutElement; // rbegin() has FNR == 0, so it will not be cutElement

    return cutElement->Boundary;
}

double TRocCurve::SelectDecisionBoundaryByIntersection() {
    CB_ENSURE(Points.size() > 0, "ROC curve must be non-empty.");
    return Points[RateCurvesIntersection].Boundary;
}

TVector<TRocPoint> TRocCurve::GetCurvePoints() {
    return Points;
}

void TRocCurve::OutputRocCurve(const TString& outputPath) {
    TFileOutput out(outputPath);
    out << "FPR" << "\t" << "TPR" << "\t" << "Threshold" << Endl;
    for (const TRocPoint& point : Points) {
        out << point.FalsePositiveRate << "\t" << 1 - point.FalseNegativeRate << "\t" << point.Boundary << Endl;
    }
}
