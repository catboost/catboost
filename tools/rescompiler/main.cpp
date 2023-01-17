#include <library/cpp/resource/registry.h>

#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/digest/city.h>
#include <util/string/cast.h>
#include <util/string/hex.h>
#include <util/string/vector.h>
#include <util/string/split.h>

using namespace NResource;

static inline void GenOne(const TString& data, const TString& key, IOutputStream& out) {
    const TString name = "name" + ToString(CityHash64(key.data(), key.size()));

    out << "static const unsigned char " << name << "[] = {";

    const TString c = Compress(data);
    char buf[16];

    for (size_t i = 0; i < c.size(); ++i) {
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

    TFixedBufferFileOutput out(argv[1]);

    argv = argv + 2;

    out << "#include <library/cpp/resource/registry.h>\n\n";

    while (*argv) {
        if ("-"sv == *argv) {
            TVector<TString> items = StringSplitter(TString(*(argv + 1))).Split('=').Limit(2).ToList<TString>();
            GenOne(TString(items[1]), TString(items[0]), out);
        } else {
            const char* key = *(argv + 1);
            if (*key == '-') {
                ++key;
            }
            GenOne(TUnbufferedFileInput(*argv).ReadAll(), key, out);
        }
        argv += 2;
    }
}
