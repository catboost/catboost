#include "delim_stroka_iter.h"

//
// TKeyValueDelimStrokaIter
//

void TKeyValueDelimStrokaIter::ReadKeyAndValue() {
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

TKeyValueDelimStrokaIter::TKeyValueDelimStrokaIter(const TStringBuf str, const TStringBuf delim)
    : DelimIter(str, delim)
{
    if (DelimIter.Valid())
        ReadKeyAndValue();
}

bool TKeyValueDelimStrokaIter::Valid() const {
    return DelimIter.Valid();
}

TKeyValueDelimStrokaIter& TKeyValueDelimStrokaIter::operator++() {
    ++DelimIter;
    if (DelimIter.Valid())
        ReadKeyAndValue();

    return *this;
}

const TStringBuf& TKeyValueDelimStrokaIter::Key() const {
    return ChunkKey;
}

const TStringBuf& TKeyValueDelimStrokaIter::Value() const {
    return ChunkValue;
}
