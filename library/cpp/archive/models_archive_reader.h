#pragma once

#include <util/generic/fwd.h>
#include <util/generic/ptr.h>
#include <util/generic/hash_set.h>

class IInputStream;

class TBlob;

class IModelsArchiveReader {
public:
    virtual ~IModelsArchiveReader() = default;
    virtual size_t Count() const = 0;
    virtual TString KeyByIndex(size_t n) const = 0;
    virtual bool Has(const TStringBuf key) const = 0;
    virtual TAutoPtr<IInputStream> ObjectByKey(const TStringBuf key) const = 0;
    virtual TBlob ObjectBlobByKey(const TStringBuf key) const = 0;
    virtual TBlob BlobByKey(const TStringBuf key) const = 0;
    virtual bool Compressed() const = 0;
    virtual THashSet<TStringBuf> FilterByPrefix(TStringBuf prefix, TStringBuf suffix) const {
        THashSet<TStringBuf> result;
        for (size_t ind = 0; ind < Count(); ++ind) { 
                TStringBuf path = KeyByIndex(ind);
                if (path.StartsWith(prefix) && path.EndsWith(suffix)) {
                    result.insert(path);
                }
        }
        return result;
    }
};
