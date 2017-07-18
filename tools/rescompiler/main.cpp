#include <library/resource/registry.h>

#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/digest/city.h>
#include <util/string/cast.h>
#include <util/string/hex.h>
#include <util/string/vector.h>
#include <util/string/iterator.h>

using namespace NResource;

static inline void GenOne(const TString& data, const TString& key, TOutputStream& out) {
    const TString name = "name" + ToString(CityHash64(~key, +key));

    out << "static const unsigned char " << name << "[] = {";

    const TString c = Compress(data);
    char buf[16];

    for (size_t i = 0; i < +c; ++i) {
        if ((i % 10) == 0) {
            out << "\n    ";
        }

        const char ch = c[i];

        out << "0x" << TStringBuf(buf, HexEncode(&ch, 1, buf)) << ", ";
    }

    out << "\n};\n\nstatic const NResource::TRegHelper REG_" << name << "(\"" << key << "\", TStringBuf((const char*)" << name << ", sizeof(" << name << ")));\n";
}

int main(int argc, char** argv) {
    if ((argc < 4) || (argc % 2)) {
        Cerr << "usage: " << argv[0] << " outfile [infile path]+ [- key=value]+" << Endl;

        return 1;
    }

    TBufferedFileOutput out(argv[1]);

    argv = argv + 2;

    out << "#include <library/resource/registry.h>\n\n";

    while (*argv) {
        if (STRINGBUF("-") == *argv) {
            yvector<TString> items = StringSplitter(TString(*(argv + 1))).SplitByStringLimited("=", 2).ToList<TString>();
            GenOne(TString(items[1]), TString(items[0]), out);
        } else {
            GenOne(TFileInput(*argv).ReadAll(), *(argv + 1), out);
        }
        argv += 2;
    }
}
