#include "log.h"
#include "element.h"

#include <utility>


TLogElement::TLogElement(const TLog* parent)
    : Parent_(parent)
    , Priority_(Parent_->DefaultPriority())
{
    Reset();
}

TLogElement::TLogElement(const TLog* parent, ELogPriority priority)
    : Parent_(parent)
    , Priority_(priority)
{
    Reset();
}

TLogElement::~TLogElement() {
    try {
        Finish();
    } catch (...) {
    }
}

void TLogElement::DoFlush() {
    if (IsNull()) {
        return;
    }

    const size_t filled = Filled();

    if (filled) {
        Parent_->Write(Priority_, Data(), filled, std::move(Context_));
        Reset();
    }
}
