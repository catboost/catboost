#include "directory_models_archive_reader.h"
#include "yarchive.h"

#include <util/folder/dirut.h>
#include <util/folder/filelist.h>
#include <util/folder/path.h>
#include <util/memory/blob.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/stream/mem.h>

TDirectoryModelsArchiveReader::TDirectoryModelsArchiveReader(const TString& path, bool lockMemory, bool ownBlobs)
    : Path_(path)
{
    Y_ENSURE(IsDir(path), "directory not found on this path");

    LoadFilesAndSubdirs("", lockMemory, ownBlobs);
}

TDirectoryModelsArchiveReader::~TDirectoryModelsArchiveReader() {}

size_t TDirectoryModelsArchiveReader::Count() const noexcept {
    return Recs_.size();
}

TString TDirectoryModelsArchiveReader::KeyByIndex(size_t n) const {
   Y_ENSURE(n < Count(), "incorrect index " << n);
   return Recs_[n];
}

bool TDirectoryModelsArchiveReader::Has(const TStringBuf key) const {
    return BlobByKey_.contains(key) || PathByKey_.contains(key);
}

namespace {
    struct TBlobOwningStream : public TMemoryInput {
        TBlob Blob;
        TBlobOwningStream(TBlob blob)
        : TMemoryInput(blob.Data(), blob.Length())
        , Blob(blob)
        {}
    };
}

TAutoPtr<IInputStream> TDirectoryModelsArchiveReader::ObjectByKey(const TStringBuf key) const {
    return new TBlobOwningStream(BlobByKey(key));
}

TBlob TDirectoryModelsArchiveReader::ObjectBlobByKey(const TStringBuf key) const {
    return BlobByKey(key);
}

TBlob TDirectoryModelsArchiveReader::BlobByKey(const TStringBuf key) const {
    Y_ENSURE(Has(key), "key " << key << " not found");
    if (auto ptr = BlobByKey_.FindPtr(key); ptr) {
        return *ptr;
    }
    if (auto ptr = PathByKey_.FindPtr(key); ptr) {
        return TBlob::FromFile(*ptr);
    }
    Y_UNREACHABLE();
}

bool TDirectoryModelsArchiveReader::Compressed() const {
    return false;
}

TString TDirectoryModelsArchiveReader::NormalizePath(TString path) const {
    path = "/" + path;
    for (size_t i = 0; i < path.size(); i++) {
        if (path[i] == '\\')
            path[i] = '/';
    }
    return path;
}

void TDirectoryModelsArchiveReader::LoadFilesAndSubdirs(const TString& subPath, bool lockMemory, bool ownBlobs) {
    TFileList fileList;
    fileList.Fill(JoinFsPaths(Path_, subPath));
    const char* file;
    while ((file = fileList.Next()) != nullptr) {
        TString key = JoinFsPaths(subPath, TString(file));
        TString fullPath = JoinFsPaths(Path_, key);
        TBlob fileBlob;
        if (lockMemory) {
            fileBlob = TBlob::LockedFromFile(fullPath);
        } else {
            fileBlob = TBlob::FromFile(fullPath);
        }
        if (key.EndsWith(".archive")) {
            TArchiveReader reader(fileBlob);
            for (size_t i = 0, iEnd = reader.Count(); i < iEnd; ++i) {
                const TString archiveKey = reader.KeyByIndex(i);
                const TString normalizedPath = NormalizePath(JoinFsPaths(subPath, archiveKey.substr(1)));
                BlobByKey_.emplace(normalizedPath, reader.ObjectBlobByKey(archiveKey));
                Recs_.push_back(normalizedPath);
            }
        } else {
            const TString normalizedPath = NormalizePath(key);
            if (lockMemory || ownBlobs) {
                BlobByKey_.emplace(normalizedPath, fileBlob);
            } else {
                PathByKey_.emplace(normalizedPath, RealPath(fullPath));
            }
            Recs_.push_back(normalizedPath);
        }
    }

    TDirsList dirsList;
    dirsList.Fill(JoinFsPaths(Path_, subPath));
    const char* dir;
    while ((dir = dirsList.Next()) != nullptr) {
        LoadFilesAndSubdirs(JoinFsPaths(subPath, TString(dir)), lockMemory, ownBlobs);
    }
}
