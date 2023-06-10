#include "csv.h"

TStringBuf NCsvFormat::CsvSplitter::Consume() {
    if (Begin == End) {
        return nullptr;
    }
    TString::iterator TokenStart = Begin;
    TString::iterator TokenEnd = Begin;
    if (Quote == '\0') {
        while (1) {
            if (TokenEnd == End || *TokenEnd == Delimeter) {
                Begin = TokenEnd;
                return TStringBuf(TokenStart, TokenEnd);
            }
            ++TokenEnd;
        }
    } else {
        bool Escape = false;
        if (*Begin == Quote) {
            Escape = true;
            ++TokenStart;
            ++TokenEnd;
            Y_ENSURE(TokenStart != End, TStringBuf("RFC4180 violation: quotation mark must be followed by something"));
        }
        while (1) {
            if (TokenEnd == End || (!Escape && *TokenEnd == Delimeter)) {
                Begin = TokenEnd;
                return TStringBuf(TokenStart, TokenEnd);
            } else if (*TokenEnd == Quote) {
                Y_ENSURE(Escape, TStringBuf("RFC4180 violation: quotation mark must be in the escaped string only"));
                if (TokenEnd + 1 == End) {
                    Begin = TokenEnd + 1;
                } else if (*(TokenEnd + 1) == Delimeter) {
                    Begin = TokenEnd + 1;
                } else if (*(TokenEnd + 1) == Quote) {
                    CustomStringBufs.push_back(TStringBuf(TokenStart, (TokenEnd + 1)));
                    TokenEnd += 2;
                    TokenStart = TokenEnd;
                    continue;
                } else {
                    Y_ENSURE(false, TStringBuf("RFC4180 violation: in escaped string quotation mark must be followed by a delimiter, EOL or another quotation mark"));
                }
                if (CustomStringBufs.size()) {
                    CustomString.clear();
                    for (auto CustomStringBuf : CustomStringBufs) {
                        CustomString += TString{ CustomStringBuf };
                    }
                    CustomString += TString{ TStringBuf(TokenStart, TokenEnd) };
                    CustomStringBufs.clear();
                    return TStringBuf(CustomString);
                } else {
                    return TStringBuf(TokenStart, TokenEnd);
                }
            }
            ++TokenEnd;
        }
    }
}

TString NCsvFormat::TLinesSplitter::ConsumeLine() {
    bool Escape = false;
    TString result;
    TString line;
    while (Input.ReadLine(line)) {
        for (auto it = line.begin(); it != line.end(); ++it) {
            if (*it == Quote) {
                Escape = !Escape;
            }
        }
        if (!result) {
            result = line;
        } else {
            result += line;
        }
        if (!Escape) {
            break;
        } else {
            result += "\n";
        }
    }
    return result;
}
