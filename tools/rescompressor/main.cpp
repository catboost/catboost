#include <library/cpp/resource/registry.h>
#include <util/stream/file.h>
#include <util/folder/path.h>

using namespace NResource;

class TAsmWriter {
private:
    IOutputStream& AsmOut;
    TString AsmPrefix;

public:
    TAsmWriter(IOutputStream& out, const TString& prefix) : AsmOut(out), AsmPrefix(prefix)
    {
    }

    void Write(TStringBuf filename, const TString& data, bool raw) {
        TString constname(Basename(filename));
        if (constname.rfind('.') != TStringBuf::npos) {
            constname = constname.substr(0, constname.rfind('.'));
        }
        WriteHeader(constname);
        if (raw) {
            WriteRaw(constname, data);
        } else {
            WriteIncBin(constname, filename, data);
        }
        WriteFooter(constname);
        WriteSymbolSize(constname);
    }

private:
    void WriteHeader(const TStringBuf& constname) {
        AsmOut << "global " << AsmPrefix << constname << "\n";
        AsmOut << "global " << AsmPrefix << constname << "Size\n";
        AsmOut << "SECTION .rodata\n";
    }

    void WriteFooter(TStringBuf constname) {
        AsmOut << AsmPrefix << constname << "Size:\n";
        AsmOut << "dd " << AsmPrefix << constname << ".end - " << AsmPrefix << constname << "\n";
    }

    void WriteSymbolSize(TStringBuf constname) {
        AsmOut << "%ifidn __OUTPUT_FORMAT__,elf64\n";
        AsmOut << "size " << AsmPrefix << constname << " " << AsmPrefix << constname << ".end - " << AsmPrefix << constname << "\n";
        AsmOut << "size " << AsmPrefix << constname << "Size 4\n";
        AsmOut << "%endif\n";
    }

    void WriteIncBin(TStringBuf constname, TStringBuf filename, const TString& data) {
        AsmOut << AsmPrefix << constname << ":\nincbin \"" << Basename(filename) << "\"\n";
        AsmOut << ".end:\n";
        TFixedBufferFileOutput out(filename.data());
        out << data;
    }

    void WriteRaw(TStringBuf constname, const TString& data) {
        AsmOut << AsmPrefix << constname << ":\ndb ";
        for (size_t i = 0; i < data.size() - 1; i++) {
            unsigned char c = static_cast<unsigned char>(data[i]);
            AsmOut << IntToString<10, unsigned char>(c) << ",";
        }
        AsmOut << IntToString<10, unsigned char>(static_cast<unsigned char>(data[data.size() - 1])) << "\n";
        AsmOut << ".end:\n";
    }

    TString Basename(TStringBuf origin) {
        TString result(origin);
        if (result.rfind('/') != TString::npos) {
            result = result.substr(result.rfind('/') + 1);
        } else if (result.rfind('\\') != TString::npos) {
            result = result.substr(result.rfind('\\') + 1);
        }
        return result;
    }
};

static TString CompressPath(const TVector<TStringBuf>& replacements, TStringBuf in) {
    for (auto r : replacements) {
        TStringBuf from, to;
        r.Split('=', from, to);
        if (in.StartsWith(from)) {
            return Compress(TString(to) + in.SubStr(from.size()));
        }
    }

    return Compress(in);
}

int main(int argc, char** argv) {
    int ind = 0;
    if (argc < 4) {
        Cerr << "usage: " << argv[ind] << "asm_output --prefix? [-? origin_resource ro_resource]+" << Endl;
        return 1;
    }

    TVector<TStringBuf> replacements;

    ind++;

    if (TStringBuf(argv[ind]) == "--compress-only") {
        ind++;
        while (ind + 1 < argc) {
            TUnbufferedFileInput inp(argv[ind]);
            TString data = inp.ReadAll();
            TString compressed = Compress(TStringBuf(data.data(), data.size()));
            TFixedBufferFileOutput out(argv[ind+1]);
            out << compressed;
            ind += 2;
        }
        return 0;
    }

    TFixedBufferFileOutput asmout(argv[ind]);
    ind++;
    TString prefix;
    if (TStringBuf(argv[ind]) == "--prefix") {
        prefix = "_";
        ind++;
    }
    else {
        prefix = "";
    }

    while (TStringBuf(argv[ind]).StartsWith("--replace=")) {
        replacements.push_back(TStringBuf(argv[ind]).SubStr(TStringBuf("--replace=").size()));
        ind++;
    }

    TAsmWriter aw(asmout, prefix);
    bool raw;
    bool error = false;
    while (ind < argc) {
        TString compressed;
        if ("-"sv == argv[ind]) {
            ind++;
            if (ind >= argc) {
                error = true;
                break;
            }
            compressed = CompressPath(replacements, TStringBuf(argv[ind]));
            raw = true;
        }
        else {
            TUnbufferedFileInput inp(argv[ind]);
            TString data = inp.ReadAll();
            compressed = Compress(TStringBuf(data.data(), data.size()));
            raw = false;
        }
        ind++;
        if (ind >= argc) {
            error = true;
            break;
        }
        aw.Write(argv[ind], compressed, raw);
        ind++;
    }
    if (error) {
        Cerr << "Incorrect number of parameters at argument " << ind - 1 << argv[ind-1] << Endl;
        return 1;
    }
    return 0;
}
