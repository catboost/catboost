#include <util/ysaveload.h>

#include "chunked_helpers.h"

TBlob GetBlock(const TBlob& blob, size_t index) {
    TChunkedDataReader reader(blob);
    if (index >= reader.GetBlocksCount())
        ythrow yexception() << "index " << index << " is >= than block count " << reader.GetBlocksCount();
    size_t begin = (const char*)reader.GetBlock(index) - (const char*)blob.Data();
    return blob.SubBlob(begin, begin + reader.GetBlockLen(index));
}

/*************************** TNamedChunkedDataReader ***************************/

static const char* NamedChunkedDataMagic = "NamedChunkedData";

TNamedChunkedDataReader::TNamedChunkedDataReader(const TBlob& blob)
    : TChunkedDataReader(blob)
{
    if (TChunkedDataReader::GetBlocksCount() < 1)
        throw yexception() << "Too few blocks";

    size_t block = TChunkedDataReader::GetBlocksCount() - 1;
    size_t magicLen = strlen(NamedChunkedDataMagic);
    if (GetBlockLen(block) < magicLen || memcmp(GetBlock(block), NamedChunkedDataMagic, magicLen) != 0)
        throw yexception() << "Not a valid named chunked data file";

    TMemoryInput input(static_cast<const char*>(GetBlock(block)) + magicLen, GetBlockLen(block) - magicLen);
    Load(&input, Names);

    size_t index = 0;
    for (TVector<TString>::const_iterator it = Names.begin(); it != Names.end(); ++it, ++index) {
        if (!it->empty())
            NameToIndex[*it] = index;
    }
}

/*************************** TNamedChunkedDataWriter ***************************/

TNamedChunkedDataWriter::TNamedChunkedDataWriter(IOutputStream& slave)
    : TChunkedDataWriter(slave)
{
}

TNamedChunkedDataWriter::~TNamedChunkedDataWriter() {
}

void TNamedChunkedDataWriter::NewBlock() {
    NewBlock("");
}

void TNamedChunkedDataWriter::NewBlock(const TString& name) {
    if (!name.empty()) {
        if (NameToIndex.count(name) != 0)
            throw yexception() << "Block name is not unique";
        NameToIndex[name] = Names.size();
    }
    Names.push_back(name);
    TChunkedDataWriter::NewBlock();
}

void TNamedChunkedDataWriter::WriteFooter() {
    NewBlock("");
    Write(NamedChunkedDataMagic);
    Save(this, Names);
    TChunkedDataWriter::WriteFooter();
}
