#include "null.h"

TNullLogBackend::TNullLogBackend() {
}

TNullLogBackend::~TNullLogBackend() {
}

void TNullLogBackend::WriteData(const TLogRecord&) {
}

void TNullLogBackend::ReopenLog() {
}
