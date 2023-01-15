#pragma once

#include "models_archive_reader.h"

#include <util/folder/path.h>
#include <util/generic/fwd.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>


class IInputStream;

class TBlob;

class TDirectoryModelsArchiveReader : public IModelsArchiveReader {
public:
    TDirectoryModelsArchiveReader(const TString& path, bool lockMemory = false, bool ownBlobs = false);
    virtual ~TDirectoryModelsArchiveReader() override;

    virtual size_t Count() const noexcept override;
    virtual TString KeyByIndex(size_t n) const override;
    virtual bool Has(const TStringBuf key) const override;
    virtual TAutoPtr<IInputStream> ObjectByKey(const TStringBuf key) const override;
    virtual TBlob ObjectBlobByKey(const TStringBuf key) const override;
    virtual TBlob BlobByKey(const TStringBuf key) const override;
    virtual bool Compressed() const override;

private:
    TString NormalizePath(TString path) const; // in archive path works unix-like path delimiter and leading slash is neccesery
    void LoadFilesAndSubdirs(const TString& subPath, bool lockMemory, bool ownBlobs);

private:
    TString Path_;
    THashMap<TString, TString> PathByKey_;
    THashMap<TString, TBlob> BlobByKey_;
    TVector<TString> Recs_;
};
