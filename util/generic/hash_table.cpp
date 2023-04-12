#include "hash_table.h"

#include <util/string/escape.h>
#include <util/string/cast.h>

const void* const _yhashtable_empty_data[] = {(void*)3, nullptr, (void*)1};

TString NPrivate::MapKeyToString(TStringBuf key) {
    constexpr size_t HASH_KEY_MAX_LENGTH = 500;
    try {
        return EscapeC(key.substr(0, HASH_KEY_MAX_LENGTH));
    } catch (...) {
        return "TStringBuf";
    }
}

TString NPrivate::MapKeyToString(unsigned short key) {
    return ToString(key);
}

TString NPrivate::MapKeyToString(short key) {
    return ToString(key);
}

TString NPrivate::MapKeyToString(unsigned int key) {
    return ToString(key);
}

TString NPrivate::MapKeyToString(int key) {
    return ToString(key);
}

TString NPrivate::MapKeyToString(unsigned long key) {
    return ToString(key);
}

TString NPrivate::MapKeyToString(long key) {
    return ToString(key);
}

TString NPrivate::MapKeyToString(unsigned long long key) {
    return ToString(key);
}

TString NPrivate::MapKeyToString(long long key) {
    return ToString(key);
}

void NPrivate::ThrowKeyNotFoundInHashTableException(const TStringBuf keyRepresentation) {
    ythrow yexception() << "Key not found in hashtable: " << keyRepresentation;
}
