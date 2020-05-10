#include "delim_string_iter.h"

//
// TKeyValueDelimStringIter
//

void TKeyValueDelimStringIter::ReadKeyAndValue() {
    TStringBuf currentToken(*DelimIter);

    size_t pos = currentToken.find('=');
    if (pos == TString::npos) {
        ChunkValue.Clear();
        ChunkKey = currentToken;
    } else {
        ChunkKey = currentToken.SubStr(0, pos);
        ChunkValue = currentToken.SubStr(pos + 1);
    }
}

TKeyValueDelimStringIter::TKeyValueDelimStringIter(const TStringBuf str, const TStringBuf delim)
    : DelimIter(str, delim)
{
    if (DelimIter.Valid())
        ReadKeyAndValue();
}

bool TKeyValueDelimStringIter::Valid() const {
    return DelimIter.Valid();
}

TKeyValueDelimStringIter& TKeyValueDelimStringIter::operator++() {
    ++DelimIter;
    if (DelimIter.Valid())
        ReadKeyAndValue();

    return *this;
}

const TStringBuf& TKeyValueDelimStringIter::Key() const {
    return ChunkKey;
}

const TStringBuf& TKeyValueDelimStringIter::Value() const {
    return ChunkValue;
}
