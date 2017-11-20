#include <library/resource/registry.h>

#include <util/digest/city.h>
#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/string/vector.h>
#include <util/string/iterator.h>


using namespace NResource;

void GenOne(const TString& raw, const TString& key, IOutputStream& out) {
    out << "extern \"C\" {\n   extern const unsigned char " << raw << "[];\n   extern const unsigned int " << raw << "Size;\n};\n";
    out << "static const NResource::TRegHelper REG_name" << ToString(CityHash64(~key, +key)) << "(\"" << key << "\", TStringBuf((const char *)" << raw << ", " << raw << "Size));\n\n";
};

int main(int argc, char** argv) {
    if (argc < 3) {
        Cerr << "usage: " << argv[0] << " outfile [key=value]+" << Endl;

        return 1;
    }

    TFixedBufferFileOutput out(argv[1]);

    argv = argv + 2;
    out << "#include <library/resource/registry.h>\n\n";

    while (*argv) {
        TVector<TString> items = StringSplitter(TString(*(argv))).SplitByStringLimited("=", 2).ToList<TString>();
        GenOne(items[0], items[1], out);
        argv++;
    }
}
