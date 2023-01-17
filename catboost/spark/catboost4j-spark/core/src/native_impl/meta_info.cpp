#include "meta_info.h"

#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/libs/helpers/exception.h>

#include "dataset_rows_reader.h"

#include <util/generic/maybe.h>
#include <util/stream/file.h>

using namespace NCB;

bool TIntermediateDataMetaInfo::HasSparseFeatures() const {
    if (HasUnknownNumberOfSparseFeatures) {
        return true;
    }

    bool hasSparseFeatures = false;
    bool hasDenseFeatures = false;
    for (const auto& featureMetaInfo : FeaturesLayout->GetExternalFeaturesMetaInfo()) {
        if (featureMetaInfo.IsAvailable) {
            CB_ENSURE(
                featureMetaInfo.Type == EFeatureType::Float,
                "Non-float features are not supported yet"
            );
            if (featureMetaInfo.IsSparse) {
                hasSparseFeatures = true;
            } else {
                hasDenseFeatures = true;
            }
        }
    }
    CB_ENSURE(
        !(hasDenseFeatures && hasSparseFeatures),
        "Datasets with mixed dense and sparse features are not supported"
    );
    return hasSparseFeatures;
}


struct TOneLineReader final : public ILineDataReader {
    explicit TOneLineReader(TMaybe<TString>&& header, const TString& singleDataLine)
        : Header(std::move(header))
        , DataLine(singleDataLine)
    {}

    ui64 GetDataLineCount(bool /*estimate*/) override {
        return 1;
    }

    TMaybe<TString> GetHeader() override {
        return Header;
    }

    bool ReadLine(TString* line, ui64* lineIdx) override {
        if (DataLineProcessed) {
            return false;
        } else {
            if (lineIdx) {
                *lineIdx = 0;
            }
            *line = std::move(DataLine);
            DataLineProcessed = true;
            return true;
        }
    }

private:
    TMaybe<TString> Header;
    TString DataLine;
    bool DataLineProcessed = false;
};


TIntermediateDataMetaInfo GetIntermediateDataMetaInfo(
    const TString& schema,
    const TString& columnDescriptionPathWithScheme,
    const TString& plainJsonParamsAsString,
    const TMaybe<TString>& dsvHeader,
    const TString& firstDataLine
) {
    TRawDatasetRowsReader rowsReader(
        schema,
        new TOneLineReader(TMaybe<TString>(dsvHeader), firstDataLine),
        columnDescriptionPathWithScheme,
        TVector<TColumn>(),
        plainJsonParamsAsString,
        dsvHeader.Defined(),
        /*blockSize*/ 1,
        /*threadCount*/ 1
    );

    rowsReader.ReadNextBlock();
    return rowsReader.GetMetaInfo();
}
