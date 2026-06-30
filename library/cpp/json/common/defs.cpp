#include "defs.h"

using namespace NJson;

TJsonCallbacks::~TJsonCallbacks() {
}

bool TJsonCallbacks::OnNull() {
    return true;
}

bool TJsonCallbacks::OnBoolean(bool) {
    return true;
}

bool TJsonCallbacks::OnInteger(long long) {
    return true;
}

bool TJsonCallbacks::OnUInteger(unsigned long long) {
    return true;
}

bool TJsonCallbacks::OnDouble(double) {
    return true;
}

bool TJsonCallbacks::OnString(const TStringBuf&) {
    return true;
}

bool TJsonCallbacks::OnOpenMap() {
    return true;
}

bool TJsonCallbacks::OnMapKey(const TStringBuf&) {
    return true;
}

bool TJsonCallbacks::OnCloseMap() {
    return true;
}

bool TJsonCallbacks::OnOpenArray() {
    return true;
}

bool TJsonCallbacks::OnCloseArray() {
    return true;
}

bool TJsonCallbacks::OnStringNoCopy(const TStringBuf& s) {
    return OnString(s);
}

bool TJsonCallbacks::OnMapKeyNoCopy(const TStringBuf& s) {
    return OnMapKey(s);
}

bool TJsonCallbacks::OnEnd() {
    return true;
}

void TJsonCallbacks::OnError(size_t off, TStringBuf reason) {
    HaveErrors = true;
    if (ThrowException) {
        ythrow TJsonException() << "JSON error at offset " << off << " (" << reason << ")";
    }
}
