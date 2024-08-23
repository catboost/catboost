#include "patch.h"

#include <library/cpp/getopt/last_getopt.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/stream/null.h>
#include <util/string/cast.h>
#include <util/system/defaults.h>

namespace NElf {

bool IsElf(const TString& path) {
    TUnbufferedFileInput in(path);
    char buffer[EI_NIDENT];
    size_t nread = in.Load(buffer, sizeof(buffer));

    if (nread != sizeof(buffer) || TStringBuf(buffer, SELFMAG) != ELFMAG) {
        Cerr << "fix_elf skip " << path << " (not an ELF file)";
        return false;
    }

    if (buffer[EI_CLASS] != ELFCLASS64) {
        Cerr << "fix_elf skip " << path << " (ELF class is not ELF64)";
        return false;
    }

#ifdef _little_endian_
    if (buffer[EI_DATA] != ELFDATA2LSB) {
        Cerr << "fix_elf skip " << path << " (ELF byte order is not native LSB)";
        return false;
    }
#else
    if (buffer[EI_DATA] != ELFDATA2MSB) {
        Cerr << "fix_elf skip " << path << " (ELF byte order is not native MSB)";
        return false;
    }
#endif

    if (buffer[EI_VERSION] != 1) {
        Cerr << "fix_elf skip " << path << " (ELF version is not 1)";
        return false;
    }

    return true;
}

} // namespace NElf

using namespace NElf;

void ReadNum(TStringBuf& src, TStringBuf& dst) {
    const char* c = src.data();
    while (isdigit(*c)) {
        ++c;
    }
    size_t len = c - src.data();

    dst = TStringBuf(src.data(), len);
    src.Skip(len);
}

int NumericStrCmp(TStringBuf s1, TStringBuf s2) {
    while (!s1.empty() || !s2.empty()) {
        char c1 = *s1.data();
        char c2 = *s2.data();

        if (isdigit(c1) && isdigit(c2)) {
            TStringBuf num1, num2;
            ReadNum(s1, num1);
            ReadNum(s2, num2);

            int c = FromString<int>(num1) - FromString<int>(num2);
            if (c) {
                return c;
            }

        } else {
            int c = int(c1) - int(c2);
            if (c) {
                return c;
            }
        }

        s1.Skip(1);
        s2.Skip(1);
    }

    return 0;
}

class TVernauxCmp {
public:
    TVernauxCmp(TSection strSect)
        : StrSect(strSect)
    {
    }

    bool operator()(Elf64_Vernaux* v1, Elf64_Vernaux* v2) {
        TStringBuf s1 = StrSect.GetStr(v1->vna_name);
        TStringBuf s2 = StrSect.GetStr(v2->vna_name);

        return NumericStrCmp(s1, s2) < 0;
    }

private:
    TSection StrSect;
};

void Patch(const TString& path, const TString& library, IOutputStream& verboseOut) {
    TElf elf(path);

    TVerneedSection verneedSect(&elf);
    if (verneedSect.IsNull()) {
        verboseOut << "No symbol versions section" << Endl;
        return;
    }

    TSection verStrings(&elf, elf.GetSection(verneedSect.GetLink()));

    TStringBuf skipFrom("GLIBC_2.14");
    TStringBuf patchFrom("GLIBC_2.2.5");

    TVector<Elf64_Vernaux*> patchAux;

    Elf64_Vernaux* patchFromAux = nullptr;

    Elf64_Verneed* verneed = verneedSect.GetFirstVerneed();
    while (verneed) {

        TStringBuf file = verStrings.GetStr(verneed->vn_file);
        verboseOut << file;

        if (file != library) {
            verboseOut << " skipped" << Endl;

        } else {
            verboseOut << Endl;

            Elf64_Vernaux* vernaux = verneedSect.GetFirstVernaux(verneed);
            while (vernaux) {

                TStringBuf name = verStrings.GetStr(vernaux->vna_name);
                verboseOut << "\t" << name;

                if (!patchFromAux && name == patchFrom) {
                    verboseOut << " taken as patch source" << Endl;
                    patchFromAux = vernaux;

                } else {

                    if (NumericStrCmp(name, skipFrom) < 0) {
                        verboseOut << " skipped" << Endl;

                    } else {
                        verboseOut << " will be patched" << Endl;
                        patchAux.push_back(vernaux);
                    }
                }
                vernaux = verneedSect.GetNextVernaux(vernaux);
            }
        }
        verneed = verneedSect.GetNextVerneed(verneed);
    }

    if (patchAux.empty()) {
        verboseOut << "Nothing to patch" << Endl;
        return;
    }

    if (!patchFromAux) {
        ythrow yexception() << path << ": no ELF64_Vernaux source to patch from";
    }

    TSection dynsymSect(&elf, elf.GetSectionByType(SHT_DYNSYM));
    TSection symstrSect(&elf, elf.GetSection(dynsymSect.GetLink()));
    TSection dynverSect(&elf, elf.GetSectionByType(SHT_GNU_versym));

    for (size_t i = 0, c = dynsymSect.GetEntryCount(); i < c; ++i) {
        Elf64_Sym* sym = dynsymSect.GetEntry<Elf64_Sym>(i);
        Elf64_Half* ver = dynverSect.GetEntry<Elf64_Half>(i);
        for (auto aux : patchAux) {
            if (*ver == aux->vna_other) {
                *ver = 0;
                verboseOut << "Symbol " << i << ": " << symstrSect.GetStr(sym->st_name)
                    << "@" << verStrings.GetStr(aux->vna_name) << " version removed" << Endl;
            }
        }
    }

    for (auto aux : patchAux) {
        TStringBuf name = verStrings.GetStr(aux->vna_name);
        aux->vna_name = patchFromAux->vna_name;
        aux->vna_hash = patchFromAux->vna_hash;
        verboseOut << "Version dependence " << name << " [" << aux->vna_other
            << "] patched from " << patchFrom << " [" << patchFromAux->vna_other << "]" << Endl;
    }
}

void PatchGnuUnique(const TString& path, IOutputStream& verboseOut) {
    TElf elf(path);

    for (Elf64_Shdr* it = elf.GetSectionBegin(), *end = elf.GetSectionEnd(); it != end; ++it) {
        if (it->sh_type == SHT_SYMTAB) {

            TSection section{&elf, it};
            verboseOut << "Found symbol section [" << section.GetName() << ']' << Endl;

            for (size_t i = 0, count = section.GetEntryCount(); i < count; ++i) {
                Elf64_Sym* symbol = section.GetEntry<Elf64_Sym>(i);
                auto& info = symbol->st_info;

                if (ELF64_ST_BIND(info) == STB_GNU_UNIQUE) {
                    verboseOut << "Found GNU unique symbol #" << i << Endl;
                    info = ELF64_ST_INFO(STB_GLOBAL, ELF64_ST_TYPE(info));
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    bool rewrite_unique = false;

    using namespace NLastGetopt;

    TOpts opts = NLastGetopt::TOpts::Default();
    opts.AddHelpOption();

    opts.AddLongOption('v', "verbose").NoArgument().StoreValue(&verbose, true);
    opts.AddLongOption('u', "rewrite-gnu-unique", "Change STB_GNU_UNIQUE to STB_GLOBAL").NoArgument().StoreValue(&rewrite_unique, true);

    opts.SetFreeArgsMin(1);
    opts.SetFreeArgTitle(0, "<file>", "File");

    TOptsParseResult res(&opts, argc, argv);
    TVector<TString> files = res.GetFreeArgs();

    IOutputStream& verboseOut = verbose ? Cout : Cnull;

    bool first = true;
    for (auto path : files) {

        if (!IsElf(path)) {
            continue;
        }

        if (!first) {
            verboseOut << Endl;
        }
        first = false;

        verboseOut << "Patching " << path << Endl;

        try {
            if (rewrite_unique) {
                PatchGnuUnique(path, verboseOut);
            } else {
                Patch(path, "libc.so.6", verboseOut);
                Patch(path, "libm.so.6", verboseOut);
            }
        } catch (const yexception& e) {
            Cerr << "Patching failed: " << e.what() << Endl;
        }
    }

    return 0;
}
