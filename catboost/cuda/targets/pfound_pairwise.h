#pragma once

#include "PFound.h"
#include "kernel/PFoundPairwise.h"
#include "QuerywiseTargetsHelper.h"
#include <quality/relev_tools/matrixnet/cuda/methods/Boosting.h>
#include <quality/cuda_lib/fill.h>
#include <quality/cuda_lib/random.h>
#include <quality/cuda_lib/transform.h>
#include <quality/cuda_lib/scan.h>
#include <quality/cuda_lib/interval.h>
#include <util/generic/vector.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>

using TQuery = TVector<int>;
using TQueries = TVector<TQuery>;
using TQueryIterator = TVector<TQuery>::const_iterator;

class TPFoundFQueriesSampler {
    const static uint MAX_GROUP_SIZE = 1024;
    mutable NPar::TLocalExecutor Executor;
    const TQueries& Queries;
    const IDeviceMapping* QueriesMapping;

    double DocTakenFraction;
    double QueryTakenFraction;

    uint ThreadCount;
    TRandom& Random;
    mutable TVector<TRandom> ThreadRandoms;

    inline uint SampledQuerySize(size_t qSize) const {
        const uint maxQSize = MAX_GROUP_SIZE - 1;
        uint sampledSize = (uint)(qSize * DocTakenFraction);
        sampledSize = sampledSize > maxQSize ? maxQSize : sampledSize;
        sampledSize = sampledSize < 2 ? (const uint)Min(2, (int)qSize) : sampledSize;
        return sampledSize;
    }

    bool TakeQuery(int qid) {
        (void)qid;

        if (QueryTakenFraction < 1.0) {
            return (Random.NextUniform() < QueryTakenFraction);
        }

        return true;
    }

    TVector<uint> QuerySizes;
    TVector<uint> QueryOffsets;
    TVector<uint> GlobalSampledQueryOffsets;
    //Warning: offset are local based on query device
    TVector<uint> SampledQueryOffsets;
    TVector<uint> SampledQuerySizes;

    mutable TVector<uint> DocIds;
    TVector<uint2> Groups;
    TVector<uint2> QueryGroups;

    std::unique_ptr<IDeviceMapping> SampledDocsMapping;
    std::unique_ptr<IDeviceMapping> SampledQueriesMapping;
    std::unique_ptr<IDeviceMapping> GroupsMapping;

    void SampleQueriesAndFillOffsets() {
        QuerySizes.clear();
        QueryOffsets.clear();
        SampledQueryOffsets.clear();
        SampledQuerySizes.clear();
        DocIds.clear();

        ForeachDevice(QueriesMapping->Devices(), [&](int dev) {
            if (QueriesMapping->Type() == MT_MIRROR && dev != 0) {
                return;
            }
            uint queriesOnDevice = (uint)QueriesMapping->SizeAt(dev);
            uint queriesOffset = (uint)QueriesMapping->OffsetAt(dev);

            uint qid = 0;
            uint deviceSize = 0;
            uint docId = 0;
            int sampledQueryOffset = 0;

            for (uint q = 0; q < queriesOnDevice; ++q) {
                const auto& query = Queries[queriesOffset + q];

                if (!TakeQuery(queriesOffset + q) || (query.size() <= 1)) {
                    docId += query.size();
                } else {
                    const uint sampledSize = SampledQuerySize(query.size());

                    SampledQueryOffsets.push_back(sampledQueryOffset);
                    SampledQuerySizes.push_back(sampledSize);

                    for (uint i = 0; i < query.size(); ++i) {
                        DocIds.push_back(docId++);
                    }

                    QuerySizes.push_back((const unsigned int)query.size());
                    sampledQueryOffset += sampledSize;
                    deviceSize += sampledSize;
                    ++qid;
                }
            }

            SampledDocsMapping->SetSizeAt(dev, deviceSize);
            SampledQueriesMapping->SetSizeAt(dev, qid);
        });

        QueryOffsets.resize(QuerySizes.size(), 0);
        GlobalSampledQueryOffsets.resize(QuerySizes.size(), 0);
        for (uint q = 1; q < QuerySizes.size(); ++q) {
            QueryOffsets[q] = QueryOffsets[q - 1] + QuerySizes[q - 1];
            GlobalSampledQueryOffsets[q] = GlobalSampledQueryOffsets[q - 1] + SampledQuerySizes[q - 1];
        }
    }

    void CreateGroupping() {
        uint maxSize = 0, maxQCount = 0;
        Groups.clear();
        QueryGroups.clear();

        ForeachDevice(SampledQueriesMapping->Devices(), [&](int dev) {
            if (SampledQueriesMapping->Type() == MT_MIRROR && dev != 0) {
                return;
            }
            uint alignedSize = 0, groupQueryCount = 0, offset = 0, devGroupSize = 0;
            uint groupx = 0;
            uint qGroupx = 0;

            const size_t deviceQueryCount = SampledQueriesMapping->SizeAt(dev);
            const size_t queryOffset = SampledQueriesMapping->OffsetAt(dev);

            for (uint q = 0; q < deviceQueryCount; ++q) {
                const uint querySize = SampledQuerySizes[q + queryOffset];
                alignedSize += querySize;
                offset += querySize;
                maxSize = Max(maxSize, querySize);
                ++groupQueryCount;

                if ((alignedSize % MAX_GROUP_SIZE &&
                     (q + 1 == deviceQueryCount || (alignedSize + SampledQuerySizes[queryOffset + q + 1]) / MAX_GROUP_SIZE > alignedSize / MAX_GROUP_SIZE)) ||
                    groupQueryCount == (MAX_GROUP_SIZE / 4 - 1))
                {
                    uint2 group;
                    group.x = groupx;
                    group.y = offset;
                    groupx = group.y;
                    Groups.push_back(group);
                    ++devGroupSize;

                    uint2 qGroup;
                    qGroup.x = qGroupx;
                    qGroup.y = q + 1;
                    qGroupx = qGroup.y;
                    QueryGroups.push_back(qGroup);
                    alignedSize += MAX_GROUP_SIZE - (alignedSize % MAX_GROUP_SIZE);
                    maxQCount = Max(groupQueryCount, maxQCount);
                    groupQueryCount = 0;
                }
            }
            GroupsMapping->SetSizeAt(dev, devGroupSize);
        });

        if (maxSize >= MAX_GROUP_SIZE) {
            ythrow yexception() << "Too big query size: " << maxSize;
        }

        if (maxQCount >= MAX_GROUP_SIZE / 4) {
            ythrow yexception() << "Too much queries in single group: " << maxQCount;
        }
    }

public:
    TPFoundFQueriesSampler(const TQueries& queries,
                           const IDeviceMapping* queriesMapping,
                           TRandom& random,
                           float takenFraction = 0.5,
                           float queryFraction = 1.0)
        : Queries(queries)
        , QueriesMapping(queriesMapping)
        , DocTakenFraction(takenFraction)
        , QueryTakenFraction(queryFraction)
        , ThreadCount(NPar::LocalExecutor().GetThreadCount())
        , Random(random)
    {
        SampledDocsMapping = TCudaBufferUtils::TransformMapping(queriesMapping, [](size_t) -> size_t {
            return 0;
        });
        SampledQueriesMapping = TCudaBufferUtils::TransformMapping(queriesMapping, [](size_t) -> size_t {
            return 0;
        });
        GroupsMapping = TCudaBufferUtils::TransformMapping(queriesMapping, [](size_t) -> size_t {
            return 0;
        });

        for (uint i = 0; i < ThreadCount; ++i) {
            ThreadRandoms.push_back(TRandom(Random.NextUniformL()));
        }

        SampleQueriesAndFillOffsets();
        CreateGroupping();
    }

    void Sample(TVector<uint>& sampledDocs) const {
        NPar::TLocallyExecutableFunction sampler = [&](int tid) {
            for (uint qid = tid; qid < QuerySizes.size(); qid += ThreadCount) {
                auto start = DocIds.begin() + QueryOffsets[qid];
                auto end = start + QuerySizes[qid];

                std::random_shuffle(start, end, ThreadRandoms[tid]);
                Copy(start, start + SampledQuerySizes[qid],
                     sampledDocs.begin() + GlobalSampledQueryOffsets[qid]);
            }
        };
        NPar::LocalExecutor().ExecRange(sampler, 0, ThreadCount, Executor.WAIT_COMPLETE);
    }

    const TVector<uint>& GetSampledQueryOffsets() const {
        return SampledQueryOffsets;
    }

    const IDeviceMapping* GetSampledDocsMapping() const {
        return SampledDocsMapping.get();
    }

    const IDeviceMapping* GetGroupsMapping() const {
        return GroupsMapping.get();
    }

    const TVector<uint>& GetSampledQuerySizes() const {
        return SampledQuerySizes;
    }

    const TVector<uint2>& GetGroups() const {
        return Groups;
    }

    const TVector<uint2>& GetQueryGroups() const {
        return QueryGroups;
    }

    void SampleQueries() {
        if (QueryTakenFraction < 1.0) {
            SampleQueriesAndFillOffsets();
            CreateGroupping();
        }
    }

    bool NeedRebuildGpuData() {
        return QueryTakenFraction < 1.0;
    }

    const IDeviceMapping* GetQueriesMapping() const {
        return SampledQueriesMapping.get();
    }
};

class TPFoundFCuda: public TPFoundTarget {
protected:
    const double TakenFraction;
    const bool Querywise;
    EDecayType DecayType;
    double Decay;
    mutable TCudaBuffer<ulong> Seeds;

    mutable TCudaBuffer<uint> SampledDocs;
    mutable TVector<uint> CpuSampledDocs;
    //data after documents sampling
    mutable TCudaBuffer<uint> QuerySizes;
    mutable TCudaBuffer<uint> NzCounts;
    mutable TCudaBuffer<uint> NzOffsets;
    mutable TCudaBuffer<uint> QueryOffsets;
    mutable TCudaBuffer<uint> Qids;
    mutable TCudaBuffer<uint> PairQueryOffsets;
    mutable TCudaBuffer<uint2> Groups;
    mutable TCudaBuffer<uint2> QueryGroups;
    //for gradient calculation
    mutable TCudaBuffer<float> PairWghts;
    const IDeviceMapping* QueriesMapping;
    TPairwiseOracleImpl<TPFoundFCuda> PFoundFOracle;
    mutable TPFoundFQueriesSampler Sampler;

    //for solve nan-problem with exp
    mutable TCudaBuffer<float> WithoutQueryMeanPoint;
    mutable TQuerywiseTargetsHelper QuerywiseHelper;
    const int ResampleCount;

    void CreateGradientBuffers(const TCudaBuffer<uint>& nzOffsets, const TCudaBuffer<uint>& nzSizes,
                               TCudaBuffer<float>& grad, TCudaBuffer<float>& weights, TCudaBuffer<uint2>& pairIndices) const {
        grad.UpdateMapping(PairWghts.GetMapping());
        weights.UpdateMapping(PairWghts.GetMapping());
        pairIndices.UpdateMapping(PairWghts.GetMapping());

        auto nzMapping = nzOffsets.CopyMapping();
        auto queriesMapping = nzOffsets.GetMapping();

        TCudaBufferUtils::Foreach(nzMapping.get(), [&](int devId) {
            if (queriesMapping->Type() == MT_MIRROR && devId != 0) {
                return;
            }
            const uint lastDevQid = (const uint)(queriesMapping->OffsetAt(devId) + queriesMapping->SizeAt(devId) - 1);
            nzMapping->SetSizeAt(devId, nzOffsets[lastDevQid] + nzSizes[lastDevQid]);
        });

        grad.UpdateMapping(nzMapping.get());
        weights.UpdateMapping(nzMapping.get());
        pairIndices.UpdateMapping(nzMapping.get());
    }

    virtual void RebuildGpuData() const {
        SampledDocs.UpdateMapping(Sampler.GetSampledDocsMapping());
        CpuSampledDocs.resize(SampledDocs.Size());

        QuerySizes.UpdateMapping(Sampler.GetQueriesMapping());
        QuerySizes.Write(Sampler.GetSampledQuerySizes());

        QueryOffsets.UpdateMapping(Sampler.GetQueriesMapping());
        QueryOffsets.Write(Sampler.GetSampledQueryOffsets());

        Qids.UpdateMapping(SampledDocs.GetMapping());

        TVector<uint> qids;
        qids.reserve(Qids.Size());

        TVector<uint> pairOffsets;
        pairOffsets.reserve(Qids.Size());

        auto pairWeightsMapping = Qids.CopyMapping();

        const auto& sampledQuerySizes = Sampler.GetSampledQuerySizes();

        const auto queriesMapping = Sampler.GetQueriesMapping();
        TCudaBufferUtils::Foreach(queriesMapping, [&](int devId) {
            if (queriesMapping->Type() == MT_MIRROR && devId != 0) {
                return;
            }
            uint queryCount = (uint)queriesMapping->SizeAt(devId);
            uint queryOffset = (uint)queriesMapping->OffsetAt(devId);
            int offset = 0;

            for (uint q = 0; q < queryCount; ++q) {
                const uint qSize = sampledQuerySizes[queryOffset + q];
                qids.resize(qids.size() + qSize, q);
                pairOffsets.push_back(offset);
                offset += qSize * (qSize - 1) / 2;
            }
            pairWeightsMapping->SetSizeAt(devId, offset);
        });

        Qids.Write(qids);

        PairQueryOffsets.UpdateMapping(Sampler.GetQueriesMapping());
        PairQueryOffsets.Write(pairOffsets);

        PairWghts.UpdateMapping(pairWeightsMapping.get());

        Groups.UpdateMapping(Sampler.GetGroupsMapping());
        Groups.Write(Sampler.GetGroups());

        QueryGroups.UpdateMapping(Groups.GetMapping());
        QueryGroups.Write(Sampler.GetQueryGroups());
    }

    inline double DocFraction() const {
        if (Querywise) {
            return 1.0;
        } else {
            return TakenFraction;
        }
    }

    inline double QueryFraction() const {
        if (!Querywise) {
            return 1.0;
        } else {
            return TakenFraction;
        }
    }

public:
    TPFoundFCuda(const TDataProvider& data,
                 const TCudaBuffer<float>& target,
                 const IDeviceMapping* queriesMapping,
                 TRandom& rnd,
                 EDecayType decayType = E_EXP_DECAY,
                 float takenFraction = 0.5,
                 bool qwise = false,
                 int resampleCount = 100,
                 double decay = 0.85)
        : TPFoundTarget(data, target, rnd)
        , TakenFraction(takenFraction)
        , Querywise(qwise)
        , DecayType(decayType)
        , Decay(decay)
        , QueriesMapping(queriesMapping)
        , PFoundFOracle(*this)
        , Sampler(data.GetQueries(), QueriesMapping, rnd, DocFraction(), QueryFraction())
        , QuerywiseHelper(*queriesMapping, *target.GetMapping(), data.GetQueries())
        , ResampleCount(resampleCount)
    {
        auto seedsMapping = TCudaBufferUtils::TransformMapping(target.GetMapping(), [&](const size_t devSize) -> size_t {
            return ((devSize + 255) / 256) * 256;
        });

        Seeds = TCudaBuffer<ulong>::Create(seedsMapping.get());
        {
            TVector<ulong> seeds(Seeds.Size());

            for (uint i = 0; i < seeds.size(); i++) {
                seeds[i] = Random().NextUniformL();
            }
            Seeds.Write(seeds);
        }
        WithoutQueryMeanPoint = TCudaBuffer<float>::CopyMapping(target);
        RebuildGpuData();
    }

    const IPairwiseOracle* GetPairwiseOracle() const override {
        return &PFoundFOracle;
    }

    void PairwiseNewtonAt(const TStripeBuffer<float>& point,
                          TStripeBuffer<float>* grad,
                          TStripeBuffer<float>* pairWeights,
                          TStripeBuffer<uint2>* pairIndices) const {
        {
            auto qidMatrixOffsets = ComputeMatrixOffset(qids);
            auto matrixMapping = ComputeMatrixSize(qidMatrixOffsets);

            TStripeBuffer<uint2> tempMatrix;
            TStripeBuffer<float> tempWeights;

            tempMatrix.Reset(matrixMappng);
            tempWeights.Reset(matrixMappng);
            PFoundPairWeights(point, qids, qidMatrixOffsets, tempMatrix, tempWeights);
            RadixSort(tempWeights, tempMatrix, false);
            auto nzMapping = GetNzMapping(tempWeights);

            pairWeights->Reset(nzMapping);
            tempWeights.Reset(nzMapping);

            pairIndices->Reset(nzMapping);
            tempMatrix.Reset(nzMapping);

            pairWeights->Copy(tempWeights);
            pairIndices->Copy(tempMatrix);
            //            LocalToGloablPairs(tempMatrix, pairIndices);
        }
        //        NCudaLib::GetCudaManager().Defragment();
    }

    void PairwiseGradientAt(const TStripeBuffer<float>& point,
                            TStripeBuffer<float>* grad,
                            TStripeBuffer<float>* pairWeights,
                            TStripeBuffer<uint2>* pairIndices) const {
        Sampler.SampleQueries();
        if (Sampler.NeedRebuildGpuData()) {
            RebuildGpuData();
        }

        //        T
        SampledDocs.Write(CpuSampledDocs);

        FillBuffer(PairWghts, 0.0f);
        auto& profiler = TCudaManager::GetProfiler();
        profiler.Start("QueryMeansInPFoundGradient");
        TCudaBufferUtils::Copy(point, WithoutQueryMeanPoint);
        QuerywiseHelper.RemoveQueryMeansFromPoint(WithoutQueryMeanPoint);
        ExpVector(WithoutQueryMeanPoint);
        profiler.Stop("QueryMeansInPFoundGradient");

        PFoundFGradientWeights(Seeds, Groups, WithoutQueryMeanPoint, GetTarget(), Qids, SampledDocs, PairQueryOffsets, PairWghts, DecayType, ResampleCount, Decay);
        NzCounts.UpdateMapping(QuerySizes.GetMapping());
        NonZeroPairCounts(QueryGroups, PairWghts, PairQueryOffsets, QuerySizes, NzCounts);

        NzOffsets.UpdateMapping(QuerySizes.GetMapping());
        ScanVector(NzCounts, NzOffsets, false);

        CreateGradientBuffers(NzOffsets, NzCounts, *pairGrad, *pairWeights, *pairIndices);

        GatherNzPairs(QueryGroups, PairWghts, PairQueryOffsets, QuerySizes, NzOffsets, *pairIndices, *pairWeights);
        FillTargetsAndDocIds(Groups, QueryGroups, QueryOffsets, WithoutQueryMeanPoint, GetTarget(),
                             SampledDocs, NzOffsets, NzCounts, *pairIndices, *pairWeights, *pairGrad);
    }
};
