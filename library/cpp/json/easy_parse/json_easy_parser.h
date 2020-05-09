#pragma once

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include "json_easy_parser_impl.h"

namespace NJson {
    /* This class filters out nodes from a source JSON by a xpath-style description. It represent these nodes as a tab-delimited string (or a vector).
     * It is useful if you need to parse a data which comes into JSON in a known and fixed format.
     * Fields are set as a list of keys separated by slash, for example:
     *    Field x/y/z in JSON { "x" : { "y" : { "w" : 1, "z" : 2 } } contains number 2.
     * In a path to a field you can also provide a special array identifier "[]", identifier of a particular field in an array (for example "[4]") or wildcard "*".
     *
     * The parser of the class supports parsing of several fields. Each of them could be marked as mandatory or as optional.
     * If a mandatory field is not found in JSON, then Parse() returns false and ConvertToTabDelimited() returns an empty string.
     * If an optional field is not found in JSON, then it's value in Parse()/ConvertToTabDelimited() is an empty string.
     * In particular ConvertToTabDelimited() always returns either an empty string, or a string of the same number of tab-delimited fields starting from the same Prefix.
     *
     * NB! Library can not extract values of not a simple type (namely it doesn't support the case when a result is a vocabulary or an array) from JSON.
     * If you expect such a case, please check json_value.h.
     */

    class TJsonParser {
        TString Prefix;

        struct TField {
            TVector<TPathElem> Path;
            bool NonEmpty;
        };
        TVector<TField> Fields;

        friend class TRewriteJsonImpl;

        void ConvertToTabDelimited(IInputStream& in, IOutputStream& out) const;

    public:
        void SetPrefix(const TString& prefix) {
            Prefix = prefix;
        }
        void AddField(const TString& path, bool mustExist);
        TString ConvertToTabDelimited(const TString& json) const;
        bool Parse(const TString& json, TVector<TString>* res) const;
    };
}
