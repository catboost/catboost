#include "xml_output.h"

#include <util/generic/xrange.h>
#include <util/string/ascii.h>
#include <util/string/builder.h>


// clang-format off
static const unsigned char IS_XML_ASCII_NAME_CHAR[128] = {
    /*       0xX0  0xX1  0xX2  0xX3  0xX4  0xX5  0xX6  0xX7  0xX8  0xX9  0xXA  0xXB  0xXC  0xXD  0xXE  0xXF */
    /*0x0X*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    /*0x1X*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    /*0x2X*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
    /*0x3X*/ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    /*0x4X*/ 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    /*0x5X*/ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
    /*0x6X*/ 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    /*0x7X*/ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
};


void CheckIsValidXmlAsciiName(TStringBuf name, TStringBuf contextForErrorMessage) {
    CB_ENSURE(!name.empty(), contextForErrorMessage << ": name is empty");

    CB_ENSURE(
        IsAscii(name[0]) && (IsAsciiAlpha(name[0]) || (name[0] == '_') || (name[0] == ':')),
        TString(contextForErrorMessage) << ": name \"" << name
            << "\" has the first character that is invalid for XML ASCII names"
    );
    for (auto i : xrange<size_t>(1, name.size())) {
        CB_ENSURE(
            IsAscii(name[i]) && IS_XML_ASCII_NAME_CHAR[(size_t)name[i]],
            TString(contextForErrorMessage) << ": name \"" << name
                << "\" has a character at code unit " << i << " that is invalid for XML ASCII names"
        );
    }
}


void WriteXmlEscaped(TStringBuf text, IOutputStream* out) {
    TStringBuilder escapedText;

    const char* unescapedBegin = text.cbegin();
    for (const char* iter = unescapedBegin; iter != text.end();) {
        auto handleEscaped = [&] (TStringBuf escapedChar) {
            escapedText << TStringBuf(unescapedBegin, iter) << escapedChar;
            unescapedBegin = ++iter;
        };

        switch (*iter) {
            case '<':
                handleEscaped("&lt;");
                break;
            case '>':
                handleEscaped("&gt;");
                break;
            case '&':
                handleEscaped("&amp;");
                break;
            case '\'':
                handleEscaped("&apos;");
                break;
            case '"':
                handleEscaped("&quot;");
                break;
            default:
                ++iter;
        }
    }

    if (escapedText.empty()) {
        (*out) << text;
    } else {
        (*out) << escapedText << TStringBuf(unescapedBegin, text.end());
    }
}


TXmlOutputContext::TXmlOutputContext(IOutputStream* out, TString rootName, TStringBuf version)
    : Out(out)
    , XmlEscapingOut(out)
{
    (*Out) << "<?xml version=\"" << version << "\" encoding=\"UTF-8\"?>\n<" << rootName;
    CurrentElement = std::move(rootName);
    CurrentElementIsEmpty = true;
}

void TXmlOutputContext::StartElement(TString localName) {
    CheckIsValidXmlAsciiName(localName, "StartElement");

    if (CurrentElementIsEmpty) {
        (*Out) << ">\n";
        Elements.push(std::move(CurrentElement));
    }
    (*Out) << '<' << localName;
    CurrentElement = std::move(localName);
    CurrentElementIsEmpty = true;
}

void TXmlOutputContext::EndElement() {
    if (CurrentElementIsEmpty) {
        (*Out) << "/>\n";
        CurrentElementIsEmpty = false;
    } else {
        (*Out) << "</" << Elements.top() << ">\n";
        Elements.pop();
    }
}

