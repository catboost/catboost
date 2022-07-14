#pragma once

#include "exception.h"

#include <util/generic/maybe.h>
#include <util/generic/noncopyable.h>
#include <util/generic/stack.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/stream/output.h>

#include <exception>
#include <type_traits>


// throws yexception is name is not a valid XML ASCII name
void CheckIsValidXmlAsciiName(TStringBuf name, TStringBuf contextForErrorMessage);

void WriteXmlEscaped(TStringBuf text, IOutputStream* out);


class TXmlEscapingStream : public IOutputStream {
public:
    explicit TXmlEscapingStream(IOutputStream* out)
        : Out(out)
    {}

    void DoWrite(const void* buf, size_t len) override {
        WriteXmlEscaped(TStringBuf((const char*)buf, len), Out);
    }

private:
    IOutputStream* Out;
};


/* STaX writer in UTF-8 encoding

   non UTF-8 element and attribute names are currently unsupported
*/
class TXmlOutputContext : public TNonCopyable {
public:
    TXmlOutputContext(IOutputStream* out, TString rootName, TStringBuf version = TStringBuf("1.0"));

    ~TXmlOutputContext() {
        if (!std::uncaught_exceptions()) {
            EndElement();
        }
    }

    void StartElement(TString localName);
    void EndElement();

    template <class TValue>
    TXmlOutputContext& AddAttr(TStringBuf name, const TValue& value) {
        CB_ENSURE(CurrentElementIsEmpty, "Adding attribute inside element body");
        CheckIsValidXmlAsciiName(name, "AddAttr");

        (*Out) << ' ' << name << "=\"";
        if constexpr(std::is_arithmetic<TValue>::value && !std::is_same<TValue, char>::value) {
            (*Out) << value;
        } else {
            WriteXmlEscaped(value, Out);
        }
        (*Out) << '\"';

        return *this;
    }

    // calling this function means adding text data inside an element
    IOutputStream& GetOutput() {
        if (CurrentElementIsEmpty) {
            (*Out) << '>';
            Elements.push(std::move(CurrentElement));
            CurrentElementIsEmpty = false;
        }
        return XmlEscapingOut;
    }

private:
    IOutputStream* Out;
    TXmlEscapingStream XmlEscapingOut;
    TString CurrentElement;
    bool CurrentElementIsEmpty;
    TStack<TString> Elements;
};


// RAII
class TXmlElementOutputContext : public TNonCopyable {
public:
    TXmlElementOutputContext(TXmlOutputContext* context, TString elementName)
        : Context(context)
    {
        Context->StartElement(std::move(elementName));
    }

    ~TXmlElementOutputContext() {
        if (!std::uncaught_exceptions()) {
            Context->EndElement();
        }
    }

private:
    TXmlOutputContext* Context;
};
