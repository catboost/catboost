#include "hash.h"

#include <util/digest/city.h>
#include <util/generic/string.h>
#include <util/memory/blob.h>
#include <util/system/file.h>
#include <util/system/fstat.h>

void ReadFile(const char* fpath, TBlob& blob) {
    TFile f(TString{fpath}, RdOnly | Seq);
    const TFileStat fs(f);
    auto size = fs.Size;

    if (size < (64 << 10)) {
        blob = TBlob::FromFileContent(f, 0, size);
    } else {
        // Read 1 byte before mapping to detect access problems for encrypted and banned files in arc
        TBlob::FromFileContentSingleThreaded(f, 0, 1);
        blob = TBlob::FromFile(f);
    }
}

ui64 FileCityHash128WithSeedHigh64(const char* fpath) {
    TBlob blob;
    ReadFile(fpath, blob);
    const uint128 hash = CityHash128WithSeed((const char*)blob.Data(), blob.Size(), uint128(0, blob.Size()));
    return Uint128High64(hash);
}

ui64 FileCityHash64(const char* fpath) {
    TBlob blob;
    ReadFile(fpath, blob);
    return CityHash64(static_cast<const char*>(blob.Data()), blob.Size());
}
