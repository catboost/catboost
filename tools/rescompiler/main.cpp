#include <library/cpp/resource/registry.h>

#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/digest/city.h>
#include <util/string/cast.h>
#include <util/string/hex.h>
#include <util/string/vector.h>
#include <util/string/split.h>

static inline void Formatter(IOutputStream& stream, std::string_view fmt, TVector<std::string_view>&& views) {
    constexpr auto INF = std::string::npos;
    size_t pos = 0;
    auto view = views.begin();
    while (pos != INF && pos < fmt.size()) {
        size_t found = fmt.find("{}", pos);
        stream << fmt.substr(pos, found == INF ? INF : found - pos);
        pos = found == INF ? INF : found + 2;
        if (view != views.end()) {
            stream << *view;
            ++view;
        }
    }
}

using namespace NResource;

static inline void EmitHex(IOutputStream& stream, const TString& view) {
    const unsigned char* data = reinterpret_cast<const unsigned char*>(view.data());
    size_t len = view.size();

    constexpr size_t CHAR2HEX = 5; // len('0x??,') == 5
    constexpr size_t MAX_STEP = 16;
    constexpr size_t STEPS[] = {MAX_STEP, 1};

    auto print = [](char* out, const unsigned char* iter, const unsigned char* end) {
        char templ[CHAR2HEX + 1] = "0x??,";
        while (iter != end) {
            HexEncode(iter, 1ULL, templ + 2ULL);
            memcpy(out, templ, CHAR2HEX);
            iter++;
            out += CHAR2HEX;
        }
    };

    char buf[CHAR2HEX * MAX_STEP + 4] = {};
    for (size_t step : STEPS) {
        while (len >= step) {
            print(buf, data, data + step);
            buf[CHAR2HEX * step] = 0;
            len -= step;
            data += step;
            stream << buf << (step > 1 ? "\n" : "");
        }
    }
    stream << '\n';
}

static inline void EmitHexArray(const TString& data, const TString& varName, IOutputStream& out) {
    const TString c = Compress(data);
    out << "static const unsigned char " << varName << "[] = {\n";
    EmitHex(out, c);
    out << "};\n";
}

static inline void GenOne(const TString& data, const TString& key, IOutputStream& out, bool isFile, bool useSections) {
    const TString name = "name" + ToString(CityHash64(key.data(), key.size()));

    if (useSections) {
        if (isFile) {
            Formatter(out, R"__(
            extern "C" const char {}[];
            extern "C" const char {}_end[];
            static const int REG_{} = NResource::LightRegisterI("{}", {}, {}_end);
            )__", {data, data, name, key, data, data});
        } else {
            EmitHexArray(data, name, out);
            Formatter(out, R"__(
            static const int REG_{} = NResource::LightRegisterS("{}", (const char*){}, sizeof({}));
            )__", {name, key, name, name});
        }
        return;
    }

    EmitHexArray(data, name, out);
    Formatter(out, R"__(
    static const NResource::TRegHelper REG_{}("{}", TStringBuf((const char*){}, sizeof({})));
    )__", {name, key, name, name});
}

static inline void EmitHeader(IOutputStream& out, bool useSections) {
    if (useSections) {
        out << R"__(
            // This function are defined in "library/cpp/resource/registry.cpp"
            namespace NResource {
                int LightRegisterS(const char*, const char*, unsigned long);
                int LightRegisterI(const char*, const char*, const char*);
            }
            )__";
    } else {
        out << "#include <library/cpp/resource/registry.h>\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        Cerr << "usage: " << argv[0] << " outfile [?--use-sections] [infile path]+ [- key=value]+" << Endl;
        return 1;
    }

    TFixedBufferFileOutput out(argv[1]);
    bool useSections = false;
    if (TStringBuf(argv[2]) == "--use-sections") {
        useSections = true;
        argv += 3;
    } else {
        argv += 2;
    }

    EmitHeader(out, useSections);

    while (*argv) {
        if ("-"sv == *argv) {
            TVector<TString> items = StringSplitter(TString(*(argv + 1))).Split('=').Limit(2).ToList<TString>();
            GenOne(TString(items[1]), TString(items[0]), out, false /*isFile*/, useSections);
        } else {
            const char* key = *(argv + 1);
            if (*key == '-') {
                ++key;
            }
            TString data = useSections ? *argv : TUnbufferedFileInput(*argv).ReadAll();
            GenOne(data, key, out, true /*isFile*/, useSections);
        }
        argv += 2;
    }
}
