#include "wrap.h"

#include <library/cpp/colorizer/colors.h>

#include <util/generic/string.h>
#include <util/stream/str.h>
#include <util/charset/utf8.h>

#include <cctype>

namespace NLastGetopt {
    TString Wrap(ui32 width, TStringBuf text, TStringBuf indent, size_t* lastLineLen, bool* hasParagraphs) {
        if (width == 0) {
            return TString(text);
        }

        if (width >= indent.size()) {
            width -= indent.size();
        }

        if (hasParagraphs) {
            *hasParagraphs = false;
        }

        TString res;
        auto os = TStringOutput(res);

        const char* spaceBegin = text.begin();
        const char* wordBegin = text.begin();
        const char* wordEnd = text.begin();
        const char* end = text.end();

        size_t lenSoFar = 0;

        bool isPreParagraph = false;

        do {
            spaceBegin = wordBegin = wordEnd;

            while (wordBegin < end && *wordBegin == ' ') {
                wordBegin++;
            }

            if (wordBegin == end) {
                break;
            }

            wordEnd = wordBegin;

            while (wordEnd < end && *wordEnd != ' ' && *wordEnd != '\n') {
                wordEnd++;
            }

            auto spaces = TStringBuf(spaceBegin, wordBegin);
            auto word = TStringBuf(wordBegin, wordEnd);

            size_t spaceLen = spaces.size();

            size_t wordLen = 0;
            if (!GetNumberOfUTF8Chars(word.data(), word.size(), wordLen)) {
                wordLen = word.size(); // not a utf8 string -- just use its binary size
            }
            wordLen -= NColorizer::TotalAnsiEscapeCodeLen(word);

            // Empty word means we've found a bunch of whitespaces followed by newline.
            // We don't want to print trailing whitespaces.
            if (word) {
                // We can't fit this word into the line -- insert additional line break.
                // We shouldn't insert line breaks if we're at the beginning of a line, hence `lenSoFar` check.
                if (lenSoFar && lenSoFar + spaceLen + wordLen > width) {
                    os << Endl << indent << word;
                    lenSoFar = wordLen;
                } else {
                    os << spaces << word;
                    lenSoFar += spaceLen + wordLen;
                }
                isPreParagraph = false;
            }

            if (wordEnd != end && *wordEnd == '\n') {
                os << Endl << indent;
                lenSoFar = 0;
                wordEnd++;
                if (hasParagraphs && isPreParagraph) {
                    *hasParagraphs = true;
                } else {
                    isPreParagraph = true;
                }
                continue;
            }
        } while (wordEnd < end);

        if (lastLineLen) {
            *lastLineLen = lenSoFar;
        }

        return res;
    }
}
