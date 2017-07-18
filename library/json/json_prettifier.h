#pragma once

#include "json_reader.h"

namespace NJson {

struct TJsonPrettifier {
    bool Unquote = false;
    ui8  Padding = 4;
    bool SingleQuotes = false;
    bool Compactify = false;
    bool Strict = false;
    bool NewUnquote = false; // use new unquote, may break old tests

    static TJsonPrettifier Prettifier(bool unquote = false, ui8 padding = 4, bool singlequotes = false) {
        TJsonPrettifier p;
        p.Unquote = unquote;
        p.Padding = padding;
        p.SingleQuotes = singlequotes;
        return p;
    }

    static TJsonPrettifier Compactifier(bool unquote = false, bool singlequote = false) {
        TJsonPrettifier p;
        p.Unquote = unquote;
        p.Padding = 0;
        p.Compactify = true;
        p.SingleQuotes = singlequote;
        return p;
    }

    bool Prettify(TStringBuf in, TOutputStream& out) const;

    TString Prettify(TStringBuf in) const;

    static bool MayUnquoteNew(TStringBuf in);
    static bool MayUnquoteOld(TStringBuf in);
};

inline TString PrettifyJson(TStringBuf in, bool unquote = false, ui8 padding = 4, bool sq = false) {
    return TJsonPrettifier::Prettifier(unquote, padding, sq).Prettify(in);
}

inline bool PrettifyJson(TStringBuf in, TOutputStream& out, bool unquote = false, ui8 padding = 4, bool sq = false) {
    return TJsonPrettifier::Prettifier(unquote, padding, sq).Prettify(in, out);
}

inline bool CompactifyJson(TStringBuf in, TOutputStream& out, bool unquote = false, bool sq = false) {
    return TJsonPrettifier::Compactifier(unquote, sq).Prettify(in, out);
}

inline TString CompactifyJson(TStringBuf in, bool unquote = false, bool sq = false) {
    return TJsonPrettifier::Compactifier(unquote, sq).Prettify(in);
}

}
