#pragma once

#include "last_getopt_opt.h"

#include <library/colorizer/fwd.h>

#include <util/generic/map.h>

namespace NLastGetopt {

enum EArgPermutation {
    REQUIRE_ORDER,
    PERMUTE,
    RETURN_IN_ORDER,
    DEFAULT_ARG_PERMUTATION = PERMUTE
};

/**
* NLastGetopt::TOpts is a storage of program options' parse rules.
* It contains information about all options, free args, some parsing options
*    and rules about interaction between options.
*
* The main point for defining program options.
*
* The parsing rules determined by the following parts:
*   - Arguments permutation. It is expected free args be after named args.
*     This point adjusts how to treat breaking this expectation.
*       if REQUIRE_ORDER is choosen, the exception during parsing will be raised,
*            the special string " -- " will be treated as end of named
*            options: all options after it will be parsed as free args
*       if PERMUTE is choosen, arguments will be rearranged in correct order,
*       if RETURN_IN_ORDER is choosen, all free args will be ommited (TODO: looks very strange)
*   - Using '+' as a prefix instead '--' for long names
*   - Using "-" as a prefix for both short and long names
*   - Allowing unknown options
*
*/
class TOpts {
    friend class TOptsParseResult;
    friend class TOptsParser;

public:
    typedef yvector<TSimpleSharedPtr<TOpt> > TOptsVector;
    TOptsVector Opts_;                                          // infomation about named (short and long) options

    EArgPermutation ArgPermutation_ = DEFAULT_ARG_PERMUTATION;  // determines how to parse positions of named and free options. See information below.
    bool AllowSingleDashForLong_    = false;                    //
    bool AllowPlusForLong_          = false;                    // using '+' instead '--' for long options

    //Allows unknwon options:
    bool AllowUnknownCharOptions_ = false;
    bool AllowUnknownLongOptions_ = false;

private:
    ui32 FreeArgsMin_; // minimal number of free args
    ui32 FreeArgsMax_; // maximal number of free args

    ymap<ui32, TFreeArgSpec> FreeArgSpecs_;                // mapping [free arg position] -> [free art specification]
    TFreeArgSpec DefaultFreeArgSpec = TFreeArgSpec("ARG"); // rule for parsing free arguments by default
    bool CustomDefaultArg_ = false;                        //true if DefaultFreeArgSpec have been reset

    TString Title;              // title of the help string
    TString CustomCmdLineDescr; // user defined help string

public:
    /**
    * Constructs TOpts from string as in getopt(3)
    */
    TOpts(const TStringBuf& optstring = TStringBuf());

    /**
    * Constructs TOpts from string as in getopt(3) and
    * additionally adds help option (for '?') and svn-verstion option (for 'V')
    */
    static TOpts Default(const TStringBuf& optstring = TStringBuf()) {
        TOpts opts(optstring);
        opts.AddHelpOption();
        opts.AddVersionOption();
        return opts;
    }

    /**
    * Checks correctness of options' descriptions.
    * Throws TConfException if validation failed.
    * Check consist of:
    *    -not intersecting of names
    *    -compability of settings, that responsable for freeArgs parsing
    */
    void Validate() const;

    /**
    * Search for the option with given long name
    * @param name     long name for search
    * @return         ptr on result (nullptr if not found)
    */
    const TOpt* FindLongOption(const TStringBuf& name) const;

    /**
    * Search for the option with given short name
    * @param c        short name for search
    * @return         ptr on result (nullptr if not found)
    */
    const TOpt* FindCharOption(char c) const;

    /**
    * Search for the option with given long name
    * @param name     long name for search
    * @return         ptr on result (nullptr if not found)
    */
    TOpt* FindLongOption(const TStringBuf& name);

    /**
    * Search for the option with given short name
    * @param c        short name for search
    * @return         ptr on result (nullptr if not found)
    */
    TOpt* FindCharOption(char c);

    /**
    * Sets title of the help string
    * @param title        title to set
    */
    void SetTitle(const TString& title) {
        Title = title;
    }

    /**
    * @return true if there is an option with given long name
    *
    * @param name        long name for search
    */
    bool HasLongOption(const TString& name) const {
        return FindLongOption(name) != nullptr;
    }

    /**
    * @return true if there is an option with given short name
    *
    * @param char        short name for search
    */
    bool HasCharOption(char c) const {
        return FindCharOption(c) != nullptr;
    }

    /**
    * Search for the option with given long name
    * @param name     long name for search
    * @return         ptr on result (nullptr if not found)
    */
    const TOpt& GetLongOption(const TStringBuf& name) const;

    /**
    * Search for the option with given long name
    * @param name     long name for search
    * @return         ptr on result (nullptr if not found)
    */
    TOpt& GetLongOption(const TStringBuf& name);

    /**
    * Search for the option with given short name
    * @param c        short name for search
    * @return         ptr on result (nullptr if not found)
    */
    const TOpt& GetCharOption(char c) const;

    /**
    * Search for the option with given short name
    * @param c        short name for search
    * @return         ptr on result (nullptr if not found)
    */
    TOpt& GetCharOption(char c);

    /**
    * @return true if short options exist
    */
    bool HasAnyShortOption() const;

    /**
    * @return true if long options exist
    */
    bool HasAnyLongOption() const;

    /**
    * Creates new [option description (TOpt)] as a copy of given one
    * @param option   source
    * @return         reference for created option
    */
    TOpt& AddOption(const TOpt& option);

    /**
    * Creates options list from string as in getopt(3)
    *
    * @param optstring   source
    */
    void AddCharOptions(const TStringBuf& optstring);

    /**
    * Creates new [option description (TOpt)] with given short name and given help string
    *
    * @param c        short name
    * @param help     help string
    * @return         reference for created option
    */
    TOpt& AddCharOption(char c, const TString& help = "") {
        return AddCharOption(c, DEFAULT_HAS_ARG, help);
    }

    /**
    * Creates new [option description (TOpt)] with given short name and given help string
    *
    * @param c        short name
    * @param help     help string
    * @return         reference for created option
    */
    TOpt& AddCharOption(char c, EHasArg hasArg, const TString& help = "") {
        TOpt option;
        option.AddShortName(c);
        option.Help(help);
        option.HasArg(hasArg);
        return AddOption(option);
    }

    /**
    * Creates new [option description (TOpt)] with given long name and given help string
    *
    * @param name     long name
    * @param help     help string
    * @return         reference for created option
    */
    TOpt& AddLongOption(const TString& name, const TString& help = "") {
        return AddLongOption(0, name, help);
    }

    /**
    * Creates new [option description (TOpt)] with given long and short names and given help string
    *
    * @param c        short name
    * @param name     long name
    * @param help     help string
    * @return         reference for created option
    */
    TOpt& AddLongOption(char c, const TString& name, const TString& help = "") {
        TOpt option;
        if (c != 0)
            option.AddShortName(c);
        option.AddLongName(name);
        option.Help(help);
        return AddOption(option);
    }

    /**
    * Creates new [option description (TOpt)] for help printing,
    *   adds appropriate handler for it
    * If "help" option already exist, will add given short name to it.
    *
    * @param c        new short name for help option
    */
    TOpt& AddHelpOption(char c = '?') {
        if (TOpt* o = FindLongOption("help")) {
            if (!o->CharIs(c))
                o->AddShortName(c);
            return *o;
        }
        return AddLongOption(c, "help", "print usage")
            .HasArg(NO_ARGUMENT)
            .Handler(&PrintUsageAndExit);
    }

    /**
    * Creates new [option description (TOpt)] for svn-revision printing,
    *   adds appropriate handler for it.
    * If "svnversion" option already exist, will add given short name to it.
    *
    * @param c        new short name for "svnversion" option
    */
    TOpt& AddVersionOption(char c = 'V') {
        if (TOpt* o = FindLongOption("svnrevision")) {
            if (!o->CharIs(c))
                o->AddShortName(c);
            return *o;
        }
        return AddLongOption(c, "svnrevision", "print svn version")
            .HasArg(NO_ARGUMENT)
            .Handler(&PrintVersionAndExit);
    }

    /**
    * Creates or finds option with given short name
    *
    * @param c        new short name for search/create
    */
    TOpt& CharOption(char c) {
        const TOpt* opt = FindCharOption(c);
        if (opt != nullptr) {
            return const_cast<TOpt&>(*opt);
        } else {
            AddCharOption(c);
            return const_cast<TOpt&>(GetCharOption(c));
        }
    }

    /**
    * @return index of option
    *
    * @param opt        pointer of option to search
    */
    size_t IndexOf(const TOpt* opt) const;

    /**
    * Replase help string with given
    *
    * @param decr        new help string
    */
    void SetCmdLineDescr(const TString& descr) {
        CustomCmdLineDescr = descr;
    }

    /**
    * Set minimal number of free args
    *
    * @param min        new value
    */
    void SetFreeArgsMin(size_t min) {
        FreeArgsMin_ = ui32(min);
    }

    /**
    * Set maximal number of free args
    *
    * @param min        new value
    */
    void SetFreeArgsMax(size_t max) {
        FreeArgsMax_ = ui32(max);
    }

    /**
    * Set exact expected number of free args
    *
    * @param count        new value
    */
    void SetFreeArgsNum(size_t count) {
        FreeArgsMin_ = ui32(count);
        FreeArgsMax_ = ui32(count);
    }

    /**
    * Set minimal and maximal number of free args
    *
    * @param min        new value for minimal
    * @param max        new value for maximal
    */
    void SetFreeArgsNum(size_t min, size_t max) {
        FreeArgsMin_ = ui32(min);
        FreeArgsMax_ = ui32(max);
    }

    /**
    * Set title and help string of free argument
    *
    * @param pos          index of argument
    * @param title        new value for argument title
    * @param help         new value for help string
    */
    void SetFreeArgTitle(size_t pos, const TString& title, const TString& help = TString());

    /**
    * Set title and help string of free argument be default
    *   (for positions, that have not individual settings)
    *
    * @param title        new value for argument title
    * @param help         new value for help string
    */
    void SetFreeArgDefaultTitle(const TString& title, const TString& help = TString());

    /**
    * Set the rule of parsing single dash as prefix of long names
    *
    * @param value     new value of the option
    */
    void SetAllowSingleDashForLong(bool value) {
        AllowSingleDashForLong_ = value;
    }

    /**
    * Print usage string
    *
    * @param program      prefix of result (path to the program)
    * @param os           destination stream
    * @param colors       colorizer
    */
    void PrintUsage(const TStringBuf& program, IOutputStream& os, const NColorizer::TColors& colors) const;

    /**
    * Print usage string
    *
    * @param program      prefix of result (path to the program)
    * @param os           destination stream
    */
    void PrintUsage(const TStringBuf& program, IOutputStream& os = Cerr) const;
private:
    /**
    * @return argument title of a free argument
    *
    * @param pos     position of the argument
    */
    const TString& GetFreeArgTitle(size_t pos) const;

    /**
    * @return argument help string of a free argument
    *
    * @param pos     position of the argument
    */
    const TString& GetFreeArgHelp(size_t pos) const;

    /**
    * Print usage helper
    *
    * @param program    prefix of result (path to the program)
    * @param os         destination stream
    * @param colors     colorizer
    */
    void PrintCmdLine(const TStringBuf& program, IOutputStream& os, const NColorizer::TColors& colors) const;

    /**
    * Print usage helper
    *
    * @param os         destination stream
    * @param colors     colorizer
    */
    void PrintFreeArgsDesc(IOutputStream& os, const NColorizer::TColors& colors) const;
};

} // NLastGetopt
