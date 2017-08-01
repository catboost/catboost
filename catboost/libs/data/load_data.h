#pragma once

#include "pool.h"
#include "column.h"
#include <string>
#include <util/string/vector.h>

struct TPoolColumnsMetaInfo {

    ui32 FactorCount;
    yvector<int> CatFeatureIds;

    ui32 BaselineCount = 0;

    bool HasQueryIds = false;
    bool HasDocIds = false;
    bool HasWeights = false;
};

class IPoolBuilder {
public:

    virtual void Start(const TPoolColumnsMetaInfo& poolMetaInfo) = 0;

    virtual void StartNextBlock(ui32 blockSize) = 0;

    virtual void AddCatFeature(ui32 localIdx, ui32 featureId, const TStringBuf& feature) = 0;

    virtual void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) = 0;

    virtual void AddTarget(ui32 localIdx, float value) = 0;

    virtual void AddWeight(ui32 localIdx, float value) = 0;

    virtual void AddQueryId(ui32 localIdx, const TStringBuf& queryId) = 0;

    virtual void AddBaseline(ui32 localIdx, ui32 offset, double value) = 0;

    virtual void AddDocId(ui32 localIdx, const TStringBuf& value) = 0;

    virtual void SetFeatureIds(const yvector<TString>& featureIds) = 0;

    virtual void Finish() = 0;

    virtual ~IPoolBuilder() = default;
};

void ReadPool(const TString& cdFile,
              const TString& poolFile,
              int threadCount,
              bool verbose,
              IPoolBuilder& poolBuilder,
              char fieldDelimiter = '\t',
              bool hasHeader = false);


void ReadPool(const TString& cdFile,
              const TString& poolFile,
              int threadCount,
              bool verbose,
              TPool* pool,
              char fieldDelimiter = '\t',
              bool hasHeader = false);

int CalcCatFeatureHash(const TStringBuf& feature);
float ConvertCatFeatureHashToFloat(int hashVal);
int ConvertFloatCatFeatureToIntHash(float feature);
