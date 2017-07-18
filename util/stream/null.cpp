#include "null.h"

#include <util/generic/singleton.h>

TNullIO& NPrivate::StdNullStream() noexcept {
    return *SingletonWithPriority<TNullIO, 4>();
}

TNullInput::TNullInput() noexcept {
}

TNullInput::~TNullInput() = default;

size_t TNullInput::DoRead(void*, size_t) {
    return 0;
}

size_t TNullInput::DoSkip(size_t) {
    return 0;
}

size_t TNullInput::DoNext(const void**, size_t) {
    return 0;
}

TNullOutput::TNullOutput() noexcept = default;

TNullOutput::~TNullOutput() = default;

void TNullOutput::DoWrite(const void* /*buf*/, size_t /*len*/) {
}

TNullIO::TNullIO() noexcept {
}

TNullIO::~TNullIO() = default;
