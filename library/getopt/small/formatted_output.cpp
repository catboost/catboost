#include "formatted_output.h"

namespace NLastGetopt {

    TFormattedOutput::IndentGuard::IndentGuard(TFormattedOutput* output)
        : Output(output)
    {
        Output->IndentLevel_ += 2;
    }

    TFormattedOutput::IndentGuard::~IndentGuard() {
        Output->IndentLevel_ -= 2;
    }

    TFormattedOutput::IndentGuard TFormattedOutput::Indent() {
        return IndentGuard(this);
    }

    TStringBuilder& TFormattedOutput::Line() {
        return Lines_.emplace_back(IndentLevel_, TStringBuilder()).second;
    }

    void TFormattedOutput::Print(IOutputStream& out) {
        for (auto&[indent, line] : Lines_) {
            if (indent && !line.empty()) {
                TTempBuf buf(indent);
                Fill(buf.Data(), buf.Data() + indent, ' ');
                out.Write(buf.Data(), indent);
            }
            out << line;
            if (!line.EndsWith('\n')) {
                out << Endl;
            }
        }
    }
}
