#pragma once

#include "models_archive_reader.h"

#include <util/generic/fwd.h>
#include <util/generic/ptr.h>


class IInputStream;
class IOutputStream;

class TBlob;

//noncompressed data will be stored with default alignment DEVTOOLS-4384
static constexpr size_t ArchiveWriterDefaultDataAlignment = 16;

class TArchiveWriter {
public:
    explicit TArchiveWriter(IOutputStream* out, bool compress = true);
    ~TArchiveWriter();

    void Flush();
    void Finish();
    void Add(const TString& key, IInputStream* src);
    void AddSynonym(const TString& existingKey, const TString& newKey);

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

class TArchiveReader : public IModelsArchiveReader {
public:
    explicit TArchiveReader(const TBlob& data);
    ~TArchiveReader() override;

    size_t Count() const noexcept override;
    TString KeyByIndex(size_t n) const override;
    bool Has(TStringBuf key) const override;
    TAutoPtr<IInputStream> ObjectByKey(TStringBuf key) const override;
    TBlob ObjectBlobByKey(TStringBuf key) const override;
    TBlob BlobByKey(TStringBuf key) const override;
    bool Compressed() const override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
