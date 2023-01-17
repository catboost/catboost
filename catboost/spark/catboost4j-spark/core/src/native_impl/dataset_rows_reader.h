#pragma once

#include "meta_info.h"

#include <catboost/libs/data/loader.h>

#include <catboost/private/libs/data_util/line_data_reader.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>
#include <util/system/yassert.h>


class TStringOutWrapper {
public:
    explicit TStringOutWrapper(TString* out)
        : Out(out)
    {}

    // for Hadoop Text format data
    void Assign(TConstArrayRef<i8> data, i32 length) {
        Out->assign((const char*)data.data(), length);
    }
private:
    TString* Out;
};

struct IJVMLineDataReader : public NCB::ILineDataReader {
    // should be marked 'final', but SWIG is unable to parse it
    bool ReadLine(TString* line, ui64* lineIdx) override {
        if (lineIdx) {
            *lineIdx = LineIndex;
        }
        ++LineIndex;
        return ReadLineJVM(TStringOutWrapper(line));
    }

    // override this method in derived JVM classes
    virtual bool ReadLineJVM(TStringOutWrapper line) = 0;

    ui64 LineIndex = 0;
};


struct TRawDatasetRow {
public:
    i64 GroupId;    // in fact ui64, but presented this way for JVM
    i32 SubgroupId; // in fact ui32, but presented this way for JVM
    i64 Timestamp;
    TString StringTarget;
    float FloatTarget;
    float Weight;
    float GroupWeight;

    TVector<i32> SparseFloatFeaturesIndices;
    TVector<double> SparseFloatFeaturesValues;

#ifndef SWIG
    // Hide from SWIG because it's can generate effective getters

    TArrayRef<double> Baselines; // points to data inside allocation for the whole block
    TArrayRef<double> DenseFloatFeatures; // points to data inside allocation for the whole block
#endif

public:
    // to avoid extra array creation in getter
    inline void GetDenseFloatFeatures(TArrayRef<double> output) const {
        Copy(DenseFloatFeatures.begin(), DenseFloatFeatures.end(), output.begin());
    }

    // to avoid extra array creation in getter
    inline void GetBaselines(TArrayRef<double> output) const {
        Copy(Baselines.begin(), Baselines.end(), output.begin());
    }
};

struct TColumn;


class TRawDatasetRowsReaderVisitor;


class TRawDatasetRowsReader {
public:
    TRawDatasetRowsReader(
        const TString& schema,

        // takes ownership
        // call swigReleaseOwnership() on IJVMLineDataReader-derived class before calling this method
        NCB::ILineDataReader* lineReader,
        const TString& columnDescriptionPathWithScheme, // if non-empty prefer it to columnDescription
        const TVector<TColumn>& columnsDescription,
        const TString& plainJsonParamsAsString,

        // specified here, not in params json because it will be different for different partitions
        bool hasHeader,
        i32 blockSize,
        i32 threadCount
    );

    TIntermediateDataMetaInfo GetMetaInfo() const {
        return MetaInfo;
    }

#ifndef SWIG
    void SetMetaInfo(TIntermediateDataMetaInfo&& dataMetaInfo) {
        MetaInfo = std::move(dataMetaInfo);
    }
#endif

    // returns block size
    i32 ReadNextBlock();

    // objectIdx means index in block
    const TRawDatasetRow& GetRow(i32 objectIdx);

private:
    NPar::TLocalExecutor LocalExecutor;
    THolder<NCB::IRawObjectsOrderDatasetLoader> Loader;

    // base class to avoid declaring full TRawDatasetRowsReaderVisitor in header
    THolder<NCB::IRawObjectsOrderDataVisitor> VisitorHolder;
    TRawDatasetRowsReaderVisitor* Visitor = nullptr;
    TIntermediateDataMetaInfo MetaInfo;
};

