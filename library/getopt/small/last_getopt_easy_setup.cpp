#include "last_getopt_easy_setup.h"

namespace NLastGetopt {
    TEasySetup::TEasySetup(const TStringBuf& optstring)
        : TOpts(optstring)
    {
        AddHelpOption();
    }

    TOpt& TEasySetup::AdjustParam(const char* longName, const char* help, const char* argName, bool required) {
        Y_ASSERT(longName);
        TOpt& o = AddLongOption(longName);
        if (help) {
            o.Help(help);
        }
        if (argName) {
            o.RequiredArgument(argName);
        } else {
            o.HasArg(NO_ARGUMENT);
        }
        if (required) {
            o.Required();
        }
        return o;
    }

    TEasySetup& TEasySetup::operator()(char shortName, const char* longName, const char* help, bool required) {
        AdjustParam(longName, help, nullptr, required).AddShortName(shortName);
        return *this;
    }

    TEasySetup& TEasySetup::operator()(char shortName, const char* longName, const char* argName, const char* help, bool required) {
        AdjustParam(longName, help, argName, required).AddShortName(shortName);
        return *this;
    }

    TEasySetup& TEasySetup::operator()(const char* longName, const char* help, bool required) {
        AdjustParam(longName, help, nullptr, required);
        return *this;
    }

    TEasySetup& TEasySetup::operator()(const char* longName, const char* argName, const char* help, bool required) {
        AdjustParam(longName, help, argName, required);
        return *this;
    }

}
