#pragma once

#include <catboost/libs/data/meta_info.h>

#include <util/generic/fwd.h>
#include <util/generic/yexception.h>

// SWIG has a separate declaration with Java wrapper methods added
#ifndef SWIG
    class TIntermediateDataMetaInfo : public NCB::TDataMetaInfo {
    public:
        TIntermediateDataMetaInfo() = default;

        TIntermediateDataMetaInfo(
            const NCB::TDataMetaInfo& dataMetaInfo,
            bool hasUnknownNumberOfSparseFeatures
        )
            : NCB::TDataMetaInfo(dataMetaInfo)
            , HasUnknownNumberOfSparseFeatures(hasUnknownNumberOfSparseFeatures)
        {}

        bool HasSparseFeatures() const;

        bool operator==(const TIntermediateDataMetaInfo& rhs) const {
            return NCB::TDataMetaInfo::EqualTo(rhs) &&
                (HasUnknownNumberOfSparseFeatures == rhs.HasUnknownNumberOfSparseFeatures);
        }

    public:
        bool HasUnknownNumberOfSparseFeatures = false;
    };
#endif


TIntermediateDataMetaInfo GetIntermediateDataMetaInfo(
    const TString& schema,
    const TString& columnDescriptionPathWithScheme, // can be empty
    const TString& plainJsonParamsAsString,
    const TMaybe<TString>& dsvHeader,
    const TString& firstDataLine
);
