#include "assertions.h"

#include <util/string/builder.h>
#include <util/string/split.h>
#include <util/system/type_name.h>

namespace NGTest::NInternal {
    namespace {
        void FormatActual(const std::exception& err, const TBackTrace* bt, TStringBuilder& out) {
            out << "an exception of type " << TypeName(err) << " "
                << "with message " << TString(err.what()).Quote() << ".";
            if (bt) {
                out << "\n   Trace: ";
                for (auto& line: StringSplitter(bt->PrintToString()).Split('\n')) {
                    out << "          " << line.Token() << "\n";
                }
            }
        }

        void FormatActual(TStringBuilder& out) {
            out << "  Actual: it throws ";
            auto exceptionPtr = std::current_exception();
            if (exceptionPtr) {
                try {
                    std::rethrow_exception(exceptionPtr);
                } catch (const yexception& err) {
                    FormatActual(err, err.BackTrace(), out);
                    return;
                } catch (const std::exception& err) {
                    FormatActual(err, nullptr, out);
                    return;
                } catch (...) {
                    out << "an unknown exception.";
                    return;
                }
            }
            out << "nothing.";
        }

        void FormatExpected(const char* statement, const char* type, const TString& contains, TStringBuilder& out) {
            out << "Expected: ";
            if (TStringBuf(statement).size() > 80) {
                out << "statement";
            } else {
                out << statement;
            }
            out << " throws an exception of type " << type;

            if (!contains.empty()) {
                out << " with message containing " << contains.Quote();
            }

            out << ".";
        }
    }

    TString FormatErrorWrongException(const char* statement, const char* type) {
        return FormatErrorWrongException(statement, type, "");
    }

    TString FormatErrorWrongException(const char* statement, const char* type, TString contains) {
        TStringBuilder out;

        FormatExpected(statement, type, contains, out);
        out << "\n";
        FormatActual(out);

        return out;
    }

    TString FormatErrorUnexpectedException(const char* statement) {
        TStringBuilder out;

        out << "Expected: ";
        if (TStringBuf(statement).size() > 80) {
            out << "statement";
        } else {
            out << statement;
        }
        out << " doesn't throw an exception.\n  ";

        FormatActual(out);

        return out;
    }

    bool ExceptionMessageContains(const std::exception& err, TString contains) {
        return TStringBuf(err.what()).Contains(contains);
    }
}
