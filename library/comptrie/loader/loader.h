#pragma once

#include <library/archive/yarchive.h>
#include <util/generic/string.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>
#include <util/memory/blob.h>

template <class TrieType, size_t N>
TrieType LoadTrieFromArchive(const TString& key,
                             const unsigned char (&data)[N],
                             bool ignoreErrors = false) {
    TArchiveReader archive(TBlob::NoCopy(data, sizeof(data)));
    if (archive.Has(key)) {
        TAutoPtr<IInputStream> trie = archive.ObjectByKey(key);
        return TrieType(TBlob::FromStream(*trie));
    }
    if (!ignoreErrors) {
        ythrow yexception() << "Resource " << key << " not found";
    }
    return TrieType();
}
