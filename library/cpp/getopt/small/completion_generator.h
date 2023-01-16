#pragma once

#include "completer.h"
#include "formatted_output.h"
#include "last_getopt_opts.h"
#include "modchooser.h"

#include <util/generic/variant.h>
#include <util/string/builder.h>

namespace NLastGetopt {
    class TCompletionGenerator {
    public:
        explicit TCompletionGenerator(const TModChooser* modChooser);
        explicit TCompletionGenerator(const TOpts* opts);
        virtual ~TCompletionGenerator() = default;

    public:
        virtual void Generate(TStringBuf command, IOutputStream& stream) = 0;

    protected:
        std::variant<const TModChooser*, const TOpts*> Options_;
    };

    class TZshCompletionGenerator: public TCompletionGenerator {
    public:
        using TCompletionGenerator::TCompletionGenerator;

    public:
        void Generate(TStringBuf command, IOutputStream& stream) override;

    private:
        static void GenerateModesCompletion(TFormattedOutput& out, const TModChooser& chooser, NComp::TCompleterManager& manager);
        static void GenerateOptsCompletion(TFormattedOutput& out, const TOpts& opts, NComp::TCompleterManager& manager);
        static void GenerateDefaultOptsCompletion(TFormattedOutput& out, NComp::TCompleterManager& manager);
        static void GenerateOptCompletion(TFormattedOutput& out, const TOpts& opts, const TOpt& opt, NComp::TCompleterManager& manager);
    };

    class TBashCompletionGenerator: public TCompletionGenerator {
    public:
        using TCompletionGenerator::TCompletionGenerator;

    public:
        void Generate(TStringBuf command, IOutputStream& stream) override;

    private:
        static void GenerateModesCompletion(TFormattedOutput& out, const TModChooser& chooser, NComp::TCompleterManager& manager, size_t level);
        static void GenerateOptsCompletion(TFormattedOutput& out, const TOpts& opts, NComp::TCompleterManager& manager, size_t level);
        static void GenerateDefaultOptsCompletion(TFormattedOutput& out, NComp::TCompleterManager& manager);
    };

    namespace NEscaping {
        /// Escape ':', '-', '=', '[', ']' for use in zsh _arguments
        TString Q(TStringBuf string);
        TString QQ(TStringBuf string);

        /// Escape colons for use in zsh _alternative and _arguments
        TString C(TStringBuf string);
        TString CC(TStringBuf string);

        /// Simple escape for use in zsh single-quoted strings
        TString S(TStringBuf string);
        TString SS(TStringBuf string);

        /// Simple escape for use in bash single-quoted strings
        TString B(TStringBuf string);
        TString BB(TStringBuf string);
    }
}
