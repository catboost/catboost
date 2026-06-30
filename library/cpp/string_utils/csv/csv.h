#pragma once

#include <util/generic/yexception.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/stream/input.h>

#include <vector>

/*
    Split string by rfc4180
*/

namespace NCsvFormat {
    class TLinesSplitter {
    private:
        IInputStream& Input;
        const char Quote;
    public:
        TLinesSplitter(IInputStream& input, const char quote = '"')
            : Input(input)
            , Quote(quote) {
        }
        TString ConsumeLine();
    };

    class CsvSplitter {
    public:
        CsvSplitter(const TString& data, const char delimeter = ',', const char quote = '"')
        // quote = '\0' ignores quoting in values and words like simple split
            : Delimeter(delimeter)
            , Quote(quote)
            , Begin(data.begin())
            , End(data.end())
        {
        }

        bool Step() {
            if (Begin == End) {
                return false;
            }
            ++Begin;
            return true;
        }

        TStringBuf Consume();
        explicit operator TVector<TString>() {
            TVector<TString> ret;

            do {
                TStringBuf buf = Consume();
                ret.push_back(TString{buf});
            } while (Step());

            return ret;
        }

    private:
        const char Delimeter;
        const char Quote;
        TString::const_iterator Begin;
        const TString::const_iterator End;
        std::vector<std::unique_ptr<TString>> TempResults; // CsvSplitter lifetime
        std::vector<TStringBuf> TempResultParts; // Single Consume() method call lifetime
    };
}
