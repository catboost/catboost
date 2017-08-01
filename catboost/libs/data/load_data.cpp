#include "load_data.h"
#include "load_helpers.h"

#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/split.h>

#include <library/grid_creator/binarization.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/digest/city.h>
#include <util/generic/set.h>
#include <util/generic/map.h>
#include <util/generic/hash_set.h>
#include <util/string/split.h>
#include <util/system/fs.h>
#include <util/system/event.h>
#include <util/stream/file.h>
#include <util/system/spinlock.h>


class TPoolBuilder : public IPoolBuilder {
public:
    explicit TPoolBuilder(TPool* pool)
            : Pool(pool) {
    }

    void Start(const TPoolColumnsMetaInfo& poolMetaInfo) override {
        Pool->Docs.clear();
        FactorCount = poolMetaInfo.FactorCount;
        BaselineCount = poolMetaInfo.BaselineCount;

        if (poolMetaInfo.HasQueryIds) {
            MATRIXNET_WARNING_LOG << "We don't support query ids currently" << Endl;
        }

        Pool->CatFeatures = poolMetaInfo.CatFeatureIds;
    }

    void StartNextBlock(ui32 blockSize) override {
        Cursor = (ui32) Pool->Docs.size();
        Pool->Docs.resize(Pool->Docs.size() + blockSize);
        for (ui32 i = Cursor; i < Pool->Docs.size(); ++i) {
            Pool->Docs[i].Factors.resize(FactorCount);
            Pool->Docs[i].Baseline.resize(BaselineCount);
        }
    }

    void AddCatFeature(ui32 localIdx, ui32 featureId, const TStringBuf& feature) override {
        int hashVal = CalcCatFeatureHash(feature);
        auto& curPart = LockedHashMapParts[hashVal & 0xff];
        with_lock(curPart.Lock){
            if (!curPart.CatFeatureHashes.has(hashVal)) {
                curPart.CatFeatureHashes[hashVal] = feature;
            }
        }
        AddFloatFeature(localIdx, featureId, ConvertCatFeatureHashToFloat(hashVal));
    }

    void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) override {
        GetLine(localIdx).Factors[featureId] = feature;
    }

    void AddTarget(ui32 localIdx, float value) override {
        GetLine(localIdx).Target = value;
    }

    void AddWeight(ui32 localIdx, float value) override {
        GetLine(localIdx).Weight = value;
    }

    void AddQueryId(ui32 localIdx, const TStringBuf& queryId) override {
        Y_UNUSED(localIdx);
        Y_UNUSED(queryId);
    }

    void AddBaseline(ui32 localIdx, ui32 offset, double value) override {
        GetLine(localIdx).Baseline[offset] = value;
    }

    void AddDocId(ui32 localIdx, const TStringBuf& value) override {
        GetLine(localIdx).Id = value;
    }

    void SetFeatureIds(const yvector<TString>& featureIds) override {
        Y_ENSURE(featureIds.size() == FactorCount, "Error: feature ids size should be equal to factor count");
        Pool->FeatureId = featureIds;
    }

    void Finish() override {
        if (!Pool->Docs.empty()) {
            for (const auto& part : LockedHashMapParts) {
                Pool->CatFeaturesHashToString.insert(part.CatFeatureHashes.begin(), part.CatFeatureHashes.end());
            }
            MATRIXNET_INFO_LOG << "Doc info sizes: " <<  Pool->Docs.size() << " " << FactorCount << Endl;
        } else {
            MATRIXNET_ERROR_LOG << "No doc info loaded" << Endl;
        }
    }

private:
    struct TLockedHashPart {
        TAdaptiveLock Lock;
        yhash<int, TString> CatFeatureHashes;
    };
    TDocInfo& GetLine(ui32 localIdx) {
        return Pool->Docs[Cursor + localIdx];
    }
    TPool* Pool;
    ui32 Cursor = 0;
    ui32 FactorCount = 0;
    ui32 BaselineCount = 0;
    std::array<TLockedHashPart, 256> LockedHashMapParts;
};


static yvector<int> GetCategFeatures(const yvector<TColumn>& columns) {
    Y_ASSERT(!columns.empty());
    yvector<int> categFeatures;
    int featureId = 0;
    for (const TColumn& column : columns) {
        switch (column.Type) {
            case EColumn::Categ:
                categFeatures.push_back(featureId);
                ++featureId;
                break;
            case EColumn::Num:
                ++featureId;
                break;
            case EColumn::Auxiliary:
            case EColumn::Target:
            case EColumn::Baseline:
            case EColumn::Weight:
            case EColumn::DocId:
            case EColumn::QueryId:
                break;
            default:
                CB_ENSURE(false, "this column type is not supported");
        }
    }
    return categFeatures;
}

int CalcCatFeatureHash(const TStringBuf& feature) {
    return CityHash64(feature) & 0xffffffff;
}

float ConvertCatFeatureHashToFloat(int hashVal) {
    return *reinterpret_cast<const float*>(&hashVal);
}

int ConvertFloatCatFeatureToIntHash(float feature) {
    return *reinterpret_cast<const int*>(&feature);
}

void ReadPool(const TString& cdFile,
              const TString& poolFile,
              int threadCount,
              bool verbose,
              TPool* pool,
              char fieldDelimiter,
              bool hasHeader) {
    TPoolBuilder builder(pool);
    ReadPool(cdFile, poolFile, threadCount, verbose, builder, fieldDelimiter, hasHeader);
}

void ReadPool(const TString& cdFile,
              const TString& poolFile,
              int threadCount,
              bool verbose,
              IPoolBuilder& poolBuilder,
              char fieldDelimiter,
              bool hasHeader) {
    if (verbose) {
        SetVerboseLogingMode();
    } else {
        SetSilentLogingMode();
    }

    CB_ENSURE(threadCount > 0);
    CB_ENSURE(NFs::Exists(TString(poolFile)), "pool file is not found " + TString(poolFile));
    const int columnsCount = ReadColumnsCount(poolFile, fieldDelimiter);

    yvector<TColumn> columnsDescription;

    if (!cdFile.empty()) {
        columnsDescription = ReadCD(cdFile, columnsCount);
    } else {
        columnsDescription.assign(columnsCount, TColumn{EColumn::Num, TString()});
        columnsDescription[0].Type = EColumn::Target;
    }

    TPoolColumnsMetaInfo poolMetaInfo;

    const int weightColumns = (const int) CountIf(columnsDescription.begin(), columnsDescription.end(), [](const auto x) -> bool  {
            return x.Type == EColumn::Weight;
        });
    CB_ENSURE(weightColumns <= 1, "Too many weight columns");
    poolMetaInfo.HasWeights = (bool) weightColumns;

    poolMetaInfo.BaselineCount = (ui32) CountIf(columnsDescription.begin(), columnsDescription.end(), [](const auto x) -> bool {
                return x.Type == EColumn::Baseline;
    });

    const int targetColumns = (const int) CountIf(columnsDescription.begin(), columnsDescription.end(), [](const auto x) -> bool {
            return x.Type == EColumn::Target;
        });
    CB_ENSURE(targetColumns <= 1, "Too many target columns");


    const int docIdColumns = (const int) CountIf(columnsDescription.begin(), columnsDescription.end(), [](const auto x) -> bool {
            return x.Type == EColumn::DocId;
        });
    CB_ENSURE(docIdColumns <= 1, "Too many DocId columns");
    poolMetaInfo.HasDocIds = (bool) docIdColumns;

    const int queryIdColumns = (const int) CountIf(columnsDescription.begin(), columnsDescription.end(), [](const auto x) -> bool {
        return x.Type == EColumn::QueryId;
    });
    CB_ENSURE(queryIdColumns <= 1, "Too many queryId columns");
    poolMetaInfo.HasQueryIds = (bool) queryIdColumns;

    poolMetaInfo.FactorCount = (const ui32) CountIf(columnsDescription.begin(), columnsDescription.end(), [](const auto x) -> bool{
        return IsFactorColumn(x.Type);
    });
    CB_ENSURE(poolMetaInfo.FactorCount > 0, "Pool should have at least one factor");
    poolMetaInfo.CatFeatureIds = GetCategFeatures(columnsDescription);

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);
    TIFStream reader(poolFile.c_str());
    poolBuilder.Start(poolMetaInfo);

    if (hasHeader) {
        TString line;
        reader.ReadLine(line);

        yvector<TStringBuf> words;
        SplitRangeTo<const char, yvector<TStringBuf>>(~line, ~line + line.size(), fieldDelimiter, &words);
        CB_ENSURE(words.ysize() == columnsDescription.ysize(), "wrong columns number in pool header");

        yvector<TString> featureIds;
        for (int i = 0; i < words.ysize(); ++i) {
            if (columnsDescription[i].Type == EColumn::Categ || columnsDescription[i].Type == EColumn::Num) {
                TString Id;
                TryFromString<TString>(words[i], Id);
                featureIds.push_back(Id);
            }
        }
        poolBuilder.SetFeatureIds(featureIds);
    }

    TAutoEvent blockReadCompletedEvent;
    yvector<TString> parseBuffer;
    yvector<TString> readBuffer;
    const size_t ReadBlockSize = 2048; // TODO(kirillovs): make this dynamically adjustable
    auto readLineBufferLambda = [&](int) {
        TString bufReadLine;
        readBuffer.clear();
        while (readBuffer.size() < ReadBlockSize && reader.ReadLine(bufReadLine)) {
            readBuffer.push_back(bufReadLine);
            bufReadLine.clear();
        }
        blockReadCompletedEvent.Signal();
    };

    readLineBufferLambda(0); // read first block in main thread
    blockReadCompletedEvent.WaitI();

    while (!readBuffer.empty()) {
        parseBuffer.swap(readBuffer);
        if (localExecutor.GetThreadCount() > 0) {
            localExecutor.Exec(readLineBufferLambda, 0, NPar::TLocalExecutor::HIGH_PRIORITY);
        } else {
            readLineBufferLambda(0);
        }
        poolBuilder.StartNextBlock((ui32) parseBuffer.size());

        auto parseFeaturesInBlock = [&](int lineIdx) {
            yvector<TStringBuf> words;
            const auto& line = parseBuffer[lineIdx];
            SplitRangeTo<const char, yvector<TStringBuf>>(~line, ~line + line.size(), fieldDelimiter, &words);
            CB_ENSURE(words.ysize() == columnsDescription.ysize(), "wrong columns number in pool line " << lineIdx + 1 << ": expected " << columnsDescription.ysize() << ", found " << words.ysize());

            ui32 featureId = 0;
            ui32 baselineIdx = 0;
            for (int i = 0; i < words.ysize(); ++i) {
                switch (columnsDescription[i].Type) {
                    case EColumn::Categ: {
                        poolBuilder.AddCatFeature(lineIdx, featureId, words[i]);
                        ++featureId;
                        break;
                    }
                    case EColumn::Num: {
                        float val;
                        CB_ENSURE(words[i] != "nan", "nan values not supported");
                        CB_ENSURE(words[i] != "", "empty values not supported");
                        CB_ENSURE(TryFromString<float>(words[i], val),
                                  "Factor in column " << i << " is declared as numeric and cannot be parsed as float. Try correcting column description file.");
                        poolBuilder.AddFloatFeature(lineIdx, featureId, val);
                        ++featureId;
                        break;
                    }
                    case EColumn::Target: {
                        CB_ENSURE(words[i] != "", "empty values not supported for target. Target should be float.");
                        poolBuilder.AddTarget(lineIdx, FromString<float>(words[i]));
                        break;
                    }
                    case EColumn::Weight: {
                        CB_ENSURE(words[i] != "", "empty values not supported for weight");
                        poolBuilder.AddWeight(lineIdx, FromString<float>(words[i]));
                        break;
                    }
                    case EColumn::Auxiliary: {
                        break;
                    }
                    case EColumn::QueryId: {
                        poolBuilder.AddQueryId(lineIdx, words[i]);
                        break;
                    }
                    case EColumn::Baseline: {
                        CB_ENSURE(words[i] != "", "empty values not supported for baseline");
                        poolBuilder.AddBaseline(lineIdx, baselineIdx, FromString<double>(words[i]));
                        ++baselineIdx;
                        break;
                    }
                    case EColumn::DocId: {
                        CB_ENSURE(words[i] != "", "empty values not supported for doc id");
                        poolBuilder.AddDocId(lineIdx, words[i]);
                        break;
                    }
                    default: {
                        CB_ENSURE(false, "wrong column type");
                    }
                }
            }
        };
        localExecutor.ExecRange(parseFeaturesInBlock, 0, parseBuffer.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
        blockReadCompletedEvent.WaitI();
    }

    DumpMemUsage("After data read");
    if (!AllOf(columnsDescription.begin(), columnsDescription.end(), [](const TColumn& column) {
        return column.Id.empty();
    })) {
        yvector<TString> featureIds;
        for (auto column : columnsDescription) {
            if (column.Type == EColumn::Categ || column.Type == EColumn::Num) {
                featureIds.push_back(column.Id);
            }
        }
        poolBuilder.SetFeatureIds(featureIds);
    }
    poolBuilder.Finish();

    SetVerboseLogingMode();
}
