#pragma once

#include "last_getopt_opts.h"

namespace NLastGetopt {
    /**
     * Wrapper for TOpts class to make the life a bit easier.
     * Usual usage:
     *   TEasySetup opts;
     *   opts('s', "server",  "MR_SERVER", "MapReduce server name in format server:port", true)
     *       ('u', "user",    "MR_USER",   "MapReduce user name", true)
     *       ('o', "output",  "MR_TABLE",  "Name of MR table which will contain results", true)
     *       ('r', "rules",   "FILE",      "Filename for .rules output file")          //!< This parameter is optional and has a required argument
     *       ('v', "version", &PrintSvnVersionAndExit0, "Print version information")   //!< Parameter with handler can't get any argument
     *       ("verbose", "Be verbose")                                                 //!< You may not specify short param name
     *
     *       NLastGetopt::TOptsParseResult r(&opts, argc, argv);
     */
    class TEasySetup: public TOpts {
    public:
        TEasySetup(const TStringBuf& optstring = TStringBuf());
        TEasySetup& operator()(char shortName, const char* longName, const char* help, bool required = false);
        TEasySetup& operator()(char shortName, const char* longName, const char* argName, const char* help, bool required = false);

        template <class TpFunc>
        TEasySetup& operator()(char shortName, const char* longName, TpFunc handler, const char* help, bool required = false) {
            AdjustParam(longName, help, nullptr, handler, required).AddShortName(shortName);
            return *this;
        }

        TEasySetup& operator()(const char* longName, const char* help, bool required = false);
        TEasySetup& operator()(const char* longName, const char* argName, const char* help, bool required = false);

        template <class TpFunc>
        TEasySetup& operator()(const char* longName, TpFunc handler, const char* help, bool required = false) {
            AdjustParam(longName, help, nullptr, handler, required);
            return *this;
        }

    private:
        TOpt& AdjustParam(const char* longName, const char* help, const char* argName, bool required);

        template <class TpFunc>
        TOpt& AdjustParam(const char* longName, const char* help, const char* argName, TpFunc handler, bool required) {
            TOpt& o = AdjustParam(longName, help, argName, required);
            o.Handler0(handler);
            return o;
        }
    };

}
