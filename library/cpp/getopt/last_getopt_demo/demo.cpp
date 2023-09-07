#include <library/cpp/getopt/last_getopt.h>
#include <library/cpp/getopt/modchooser.h>
#include <library/cpp/colorizer/colors.h>

// For the sake of this example, let's implement Wget

Y_COMPLETER(HeaderCompleter) {
    AddCompletion("Host");
    AddCompletion("Referer");

    bool addPostHeaders = false;

    for (int i = 0; i < argc; ++i) {
        if (argv[i] == TStringBuf("--method") && i + 1 < argc) {
            auto method = TString(argv[i + 1]);
            method.to_upper();
            addPostHeaders = method == "POST" || method == "PUT";
            break;
        } else if (argv[i] == TStringBuf("--post-data") || argv[i] == TStringBuf("--post-file")) {
            addPostHeaders = true;
            break;
        }
    }

    if (addPostHeaders) {
        AddCompletion("Content-Type");
        AddCompletion("Content-Encoding");
        AddCompletion("Content-Language");
        AddCompletion("Content-Length");
        AddCompletion("Content-Location");
        AddCompletion("Content-MD5");
        AddCompletion("Content-Range");
    }
}

class TMain: public TMainClassArgs {
    bool Background_;
    size_t Timeout_;
    TString ExplicitMethod_;
    TString ImplicitMethod_ = "GET";
    TString UserAgent_;
    TMaybe<TString> PostData_;
    TMaybe<TString> PostFile_;
    TVector<TString> Headers_;

protected:
    void RegisterOptions(NLastGetopt::TOpts& opts) override {
        // Brief description for the whole program, will appear in the beginning of a help message.
        opts.SetTitle("last_getopt_demo -- like wget, but doesn't actually do anything");

        // Built-in options.
        opts.AddHelpOption('h');
        opts.AddCompletionOption("last_getopt_demo");

        // Custom options.

        opts.AddLongOption('V', "version")
            .Help("print version and exit")
            .IfPresentDisableCompletion()
            .NoArgument()
            .Handler([]() {
                Cerr << "last_getopt_demo 1.0.0" << Endl;
                exit(0);
            });

        opts.AddLongOption('b', "background")
            .Help("go to background immediately after startup")
            .StoreTrue(&Background_);

        opts.AddLongOption("timeout")
            .RequiredArgument("timeout")
            .DefaultValue("60000")
            .Help("specify timeout in milliseconds for each request")
            .CompletionHelp("specify timeout for each request")
            .CompletionArgHelp("timeout (ms)")
            .StoreResult(&Timeout_)
            .Completer(NLastGetopt::NComp::Choice({{"1000"}, {"5000"}, {"10000"}, {"60000"}}));

        opts.AddLongOption("method")
            .RequiredArgument("http-method")
            .Help("specify HTTP method")
            .CompletionArgHelp("http method")
            .StoreResult(&ExplicitMethod_)
            .ChoicesWithCompletion({
                {"GET", "request representation of the specified resource"},
                {"HEAD", "request response identical to that of GET, but without response body"},
                {"POST", "submit an entry to the specified resource"},
                {"PUT", "replace representation of the specified resource with the request body"},
                {"DELETE", "delete the specified resource"},
                {"CONNECT", "establish a tunnel to the server identified by the target resource"},
                {"OPTIONS", "describe the communication options for the target resource"},
                {"TRACE", "perform a message loop-back test"},
                {"PATCH", "apply partial modifications to the specified resource"}});

        opts.AddLongOption('U', "user-agent")
            .RequiredArgument("agent-string")
            .DefaultValue("LastGetoptDemo/1.0.0")
            .Help("identify as `agent-string` to the HTTP server")
            .CompletionHelp("set custom user agent for each HTTP request")
            .CompletionArgHelp("user agent string")
            .StoreResult(&UserAgent_);

        opts.AddLongOption("post-data")
            .RequiredArgument("string")
            .Help("use POST method and send the specified data in the request body (cannot be used with --post-file)")
            .CompletionHelp("use POST method and send the specified data in the request body")
            .CompletionArgHelp("POST data string")
            .StoreResultT<TString>(&PostData_)
            .Handler0([this]() {
                ImplicitMethod_ = "POST";
            });

        opts.AddLongOption("post-file")
            .RequiredArgument("file")
            .Help("use POST method and send contents of the specified file in the request body (cannot be used with --post-data)")
            .CompletionHelp("use POST method and send contents of the specified file in the request body")
            .CompletionArgHelp("POST file")
            .StoreResultT<TString>(&PostFile_)
            .Handler0([this]() {
                ImplicitMethod_ = "POST";
            })
            .Completer(NLastGetopt::NComp::File());

        // These two options can't be together.
        opts.MutuallyExclusive("post-file", "post-data");

        opts.AddLongOption("header")
            .RequiredArgument("header-line")
            .Help("send `header-line` along with the rest of the headers in each HTTP request")
            .CompletionHelp("add header to each HTTP request")
            .CompletionArgHelp("header string")
            .AppendTo(&Headers_)
            .AllowMultipleCompletion()
            .Completer(NLastGetopt::NComp::LaunchSelf(HeaderCompleter));

        // Setting up free arguments.

        // We are going to have one mandatory argument and unlimited number of optional arguments.
        opts.SetFreeArgsMin(1);
        opts.SetFreeArgsMax(NLastGetopt::TOpts::UNLIMITED_ARGS);

        // Configuration for the first argument.
        opts.GetFreeArgSpec(0)
            .Title("URL")
            .Help("URL for download")
            .CompletionArgHelp("URL for download")
            .Completer(NLastGetopt::NComp::Url());

        // Configuration for optional arguments.
        opts.GetTrailingArgSpec()
            .Title("URL")
            .CompletionArgHelp("URL for download")
            .Completer(NLastGetopt::NComp::Url());

        // Let's add more text to our help. A nice description and examples.

        opts.AddSection(
            "Description",

            "LastGetoptDemo is a showcase of library/cpp/getopt capabilities. It mimics interface of Wget "
            "but doesn't actually do anything."
            "\n\n"
            "GNU Wget, on the other hand, is a free utility for non-interactive download of files from the Web."
            "It supports HTTP, HTTPS, and FTP protocols, as well as retrieval through HTTP proxies."
            "\n\n"
            "Wget is non-interactive, meaning that it can work in the background, while the user is not logged on. "
            "This allows you to start a retrieval and disconnect from the system, letting Wget finish the work. "
            "By contrast, most of the Web browsers require constant user's presence, "
            "which can be a great hindrance when transferring a lot of data."
            "\n\n"
            "Wget can follow links in HTML, XHTML, and CSS pages, to create local versions of remote web sites, "
            "fully recreating the directory structure of the original site. "
            "This is sometimes referred to as \"recursive downloading.\"  "
            "While doing that, Wget respects the Robot Exclusion Standard (/robots.txt). "
            "Wget can be instructed to convert the links in downloaded files to point at the local files, "
            "for offline viewing."
            "\n\n"
            "Wget has been designed for robustness over slow or unstable network connections; "
            "if a download fails due to a network problem, "
            "it will keep retrying until the whole file has been retrieved. "
            "If the server supports regetting, "
            "it will instruct the server to continue the download from where it left off."
            "\n\n"
            "Wget does not support Client Revocation Lists (CRLs) so the HTTPS certificate "
            "you are connecting to might be revoked by the siteowner.");

        // We will use colors for this one.
        auto& colors = NColorizer::StdErr();
        opts.AddSection(
            "Examples",

            TStringBuilder()
                << "Download a file:"
                << "\n"
                << colors.Cyan()
                << "    $ last_getopt_demo https://wordpress.org/latest.zip"
                << colors.Reset()
                << "\n"
                << "Download a file in background, set custom user agent:"
                << "\n"
                << colors.Cyan()
                << "    $ last_getopt_demo -b -U 'Wget/1.0.0' https://wordpress.org/latest.zip"
                << colors.Reset());
    }

    int DoRun(NLastGetopt::TOptsParseResult&& parsedOptions) override {
        using namespace NColorizer;

        TString method = ExplicitMethod_ ? ExplicitMethod_ : ImplicitMethod_;

        Cerr << ST_LIGHT << "Settings:" << RESET << Endl;
        Cerr << GREEN << "  Background: " << RESET << Background_ << Endl;
        Cerr << GREEN << "  Timeout: " << RESET << Timeout_ << Endl;
        Cerr << GREEN << "  Method: " << RESET << method.Quote() << Endl;
        Cerr << GREEN << "  UserAgent: " << RESET << UserAgent_.Quote() << Endl;
        Cerr << GREEN << "  PostData: " << RESET << (PostData_ ? PostData_->Quote() : "Nothing") << Endl;
        Cerr << GREEN << "  PostFile: " << RESET << (PostFile_ ? PostFile_->Quote() : "Nothing") << Endl;

        Cerr << ST_LIGHT << "Headers:" << RESET << Endl;
        for (auto& header : Headers_) {
            Cerr << "  " << header.Quote() << Endl;
        }
        if (!Headers_) {
            Cerr << GREEN << "  no headers" << RESET << Endl;
        }

        Cerr << ST_LIGHT << "Will download the following URLs:" << RESET << Endl;
        for (auto& arg : parsedOptions.GetFreeArgs()) {
            Cerr << "  " << arg.Quote() << Endl;
        }
        if (!parsedOptions.GetFreeArgs()) {
            Cerr << "  no urls" << Endl;
        }
        return 0;
    }
};

int main(int argc, const char** argv) {
    NLastGetopt::NComp::TCustomCompleter::FireCustomCompleter(argc, argv);
    TMain().Run(argc, argv);
}
