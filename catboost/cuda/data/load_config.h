#pragma once

#include <util/generic/string.h>
namespace NCatboostCuda
{
    class TPoolLoadOptions
    {
    public:
        TPoolLoadOptions() = default;

        const TString& GetFeaturesFilename() const
        {
            return FeaturesName;
        }

        const TString& GetTestFilename() const
        {
            return TestName;
        }

        const TString& GetColumnDescriptionName() const
        {
            return ColumnDescriptionName;
        }

        char GetDelimiter() const
        {
            return Delimiter;
        }

        bool HasHeader() const
        {
            return HasHeaderFlag;
        }

        const yvector<TString>& GetClassNames() const
        {
            return ClassNames;
        }

        template<class TConfig>
        friend
        class TOptionsBinder;

    private:
        TString FeaturesName = "features.tsv";
        TString TestName = "";
        TString ColumnDescriptionName = "features.cd";
        yvector<TString> ClassNames;
        char Delimiter = '\t';
        bool HasHeaderFlag = false;
    };
}
