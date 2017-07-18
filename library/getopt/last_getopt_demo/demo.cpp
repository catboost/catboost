#include <library/getopt/last_getopt.h>

#include <util/folder/dirut.h>
#include <util/generic/map.h>
#include <util/generic/string.h>
#include <util/stream/output.h>
#include <util/string/vector.h>

static void PrintVersionAndExit() {
    Cerr << "some version" << Endl;
    exit(0);
}

static TString tmpDir;

int main(int argc, char** argv) {
    int limit;
    bool longopt;
    bool extract;
    ymap<TString, TString> keyval;

    using namespace NLastGetopt;
    TOpts opts = NLastGetopt::TOpts::Default();
    opts.AddLongOption("version", "print program version").Handler(&::PrintVersionAndExit).NoArgument();
    opts.AddHelpOption();
    opts.AddLongOption("tmpdir").StoreResult(&tmpDir).DefaultValue(GetSystemTempDir()).RequiredArgument("DIR");
    opts.AddLongOption("sort").OptionalArgument("COLUMN");
    opts.AddLongOption("homedir").RequiredArgument("DIR").Required();

    opts.AddLongOption('t', "test-very-long-option").StoreResult(&longopt).DefaultValue("no")
        .Help("Run test suite for given parameters.\n\n"
              "Exit code will be non-empty if something fails.\n"
              "Please note that all extra linebreaks are ignored.\n\n\n");

    opts.AddLongOption('x', "extract-file-names").StoreResult(&extract).DefaultValue("yes")
        .Help("Extract file names while processing and print them to stdout");

    opts.AddLongOption("limit").StoreResult(&limit).DefaultValue("200")
        .Help("Non-self-descriptive and very complicated option help text.\n"
              "It was so long that we decided to format it into two likes.\n"
              "Please note that beautiful padding is preserved for multiline helps.");

    opts.AddLongOption("page-size").DefaultValue("10");

    opts.AddLongOption("set").KVHandler([&keyval](TString key, TString value) { keyval[key] = value; })
        .RequiredArgument("KEY=VALUE").Help("Set variable KEY to VALUE");

    opts.SetFreeArgsMin(1);
    opts.SetFreeArgsMax(3);

    opts.SetFreeArgTitle(0, "<aux-param>", "Auxillary parameter in free form");
    opts.SetFreeArgTitle(1, "<thread-count>"); // self-descriptive option

    TOptsParseResult res(&opts, argc, argv);

    yvector<TString> freeArgs = res.GetFreeArgs();

    ui32 pageSize = res.Get<ui32>("page-size");

    Cerr << "tmpDir:        " << tmpDir << Endl;
    Cerr << "sort:          " << res.GetOrElse("sort", "undefined") << Endl;
    Cerr << "limit:         " << limit << Endl;
    Cerr << "page size:     " << pageSize << Endl;
    Cerr << "test long opt: " << longopt << Endl;
    Cerr << "extract fnames:" << extract << Endl;
    Cerr << "free args are: [" << JoinStrings(freeArgs, ", ") << "]" << Endl;

    Cerr << "keyval {" << Endl;
    for (auto& iter: keyval) {
        Cerr << "\t" << iter.first << " : " << iter.second << Endl;
    }
    Cerr << "}" << Endl;

    return 0;
}
