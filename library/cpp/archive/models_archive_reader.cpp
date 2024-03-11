#include "models_archive_reader.h"

#include <util/generic/hash_set.h>

THashSet<TStringBuf> IModelsArchiveReader::FilterByPrefix(TStringBuf prefix, TStringBuf suffix) const {
    THashSet<TStringBuf> result;
    const size_t count = Count();
    for (size_t ind = 0; ind < count; ++ind) {
        TString path = KeyByIndex(ind);
        if (path.StartsWith(prefix) && path.EndsWith(suffix)) {
            result.insert(std::move(path));
        }
    }
    return result;
}
