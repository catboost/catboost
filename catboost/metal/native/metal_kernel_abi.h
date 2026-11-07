#pragma once
#include <cstdint>
#include <cstddef>

// Increment whenever a shared parameter field, buffer binding, winner writer,
// or reducer meaning changes. Each consuming runtime must explicitly assert
// the version it reviewed; a new version intentionally breaks stale consumers.
inline constexpr uint32_t CBMMetalKernelAbiVersion = 2;

// Matches KernelParams in CBMMetalSource. Version2 assigns byte92 to the
// weighted score before the current split, replacing a former reserved slot.
struct CBMMetalKernelParams {
    uint32_t Rows, Features, Bins, Leaves;
    uint32_t Candidates, SplitFeature, SplitBin, SplitLevel;
    float Bias, LearningRate, L2;
    uint32_t ScoreFunction;
    uint32_t Objective, LeafIteration, LeafIterations, TileRows;
    float TotalWeight;
    uint32_t HistogramTiles, PartitionTiles, ScoreGroups;
    float ObjectiveParam;
    uint32_t LeafMethod, Reserved3;
    float ScoreBeforeSplit;
};

// Version2: Score is the CTR-weighted raw score; Gain is the comparison key.
// Every writer must initialize Gain, and reducers compare Gain then Index.
struct CBMMetalSplitState {
    uint32_t Index, Feature, Bin, Type;
    float Score;
    uint32_t Valid, InvalidScore;
    float Gain;
};
static_assert(sizeof(CBMMetalKernelParams) == 96 && offsetof(CBMMetalKernelParams, ScoreBeforeSplit) == 92,
              "Shared Metal parameter ABI mismatch");
static_assert(sizeof(CBMMetalSplitState) == 32 && offsetof(CBMMetalSplitState, Gain) == 28,
              "Shared Metal winner ABI mismatch");
