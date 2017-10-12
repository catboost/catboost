#pragma once

#include <util/generic/string.h>

class TPoolLoadOptions {
public:
    TPoolLoadOptions() {
    }

    const TString& GetFeaturesFilename() const {
        return FeaturesName;
    }

    const TString& GetTestFilename() const {
        return TestName;
    }

    const TString& GetColumnDescriptionName() const {
        return ColumnDescriptionName;
    }

    char GetDelimiter() const {
        return Delimiter;
    }

    bool HasHeader() const {
        return HasHeaderFlag;
    }

    const yvector<TString>& GetClassNames() const {
        return ClassNames;
    }

    const yset<ui32>& GetIgnoredFeatures() const {
        return IgnoredFeatures;
    }

    template <class TConfig>
    friend class TOptionsBinder;

private:
    TString FeaturesName = "features.tsv";
    TString TestName = "";
    TString ColumnDescriptionName = "features.cd";
    yvector<TString> ClassNames;
    yset<ui32> IgnoredFeatures;
    char Delimiter = '\t';
    bool HasHeaderFlag = false;
};
