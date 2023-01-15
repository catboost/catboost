#include <library/cpp/resource/registry.h>

#include <util/digest/city.h>
#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/string/escape.h>
#include <util/string/vector.h>
#include <util/string/split.h>

using namespace NResource;

void GenOne(const TString& raw, const TString& key, IOutputStream& out) {
    TString size = raw + "Size";
    TString name = ToString(CityHash64(key.data(), key.size()));
    out << "extern \"C\" const char " << raw << "[];\n"
        << "extern \"C\" const unsigned int " << size << ";\n"
        << "static const NResource::TRegHelper REG_name" << name
        << "(\"" << EscapeC(key) << "\", TStringBuf(" << raw << ", " << size << "));\n"
        << "\n";
};

int main(int argc, char** argv) {
    if (argc < 3) {
        Cerr << "usage: " << argv[0] << " outfile [key=value]+" << Endl;

        return 1;
    }

    TFixedBufferFileOutput out(argv[1]);

    argv = argv + 2;
    out << "#include <library/cpp/resource/registry.h>\n\n";

    while (*argv) {
        TVector<TString> items = StringSplitter(TString(*(argv))).Split('=').Limit(2).ToList<TString>();
        GenOne(items[0], items[1], out);
        argv++;
    }
}
