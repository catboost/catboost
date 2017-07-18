#pragma once

#include <elf.h>

#include <util/generic/string.h>
#include <util/stream/file.h>
#include <util/system/filemap.h>

namespace NElf {

template<typename TTo, typename TFrom>
inline TTo Offset(TFrom from, size_t off) {
    return reinterpret_cast<TTo>(reinterpret_cast<char*>(from) + off);
}

extern const TStringBuf Magic;

bool IsElf(const TString& path);

class TElf {
public:
    TElf(const TString& path)
        : Map(path, TFileMap::oRdWr)
    {
        Map.Map(0, Map.Length());
        Begin = reinterpret_cast<char*>(Map.Ptr());

        if (Map.Length() < static_cast<i64>(sizeof(Elf64_Ehdr)) || TStringBuf(Begin, Magic.Size()) != Magic) {
            ythrow yexception() << path << " is not an ELF file";
        }
    }

    Elf64_Ehdr* GetHeader() const throw () {
        return reinterpret_cast<Elf64_Ehdr*>(Begin);
    }

    char* GetPtr(size_t offset = 0) const throw () {
        return Begin + offset;
    }

    Elf64_Shdr* GetSectionByType(Elf64_Word type) const {
        Elf64_Shdr* r = nullptr;

        for (Elf64_Shdr* p = GetSectionBegin(), *end = GetSectionEnd(); p != end; ++p) {
            if (p->sh_type == type) {
                if (r) {
                    ythrow yexception() << "More than one section of type " << type << Endl;
                }

                r = p;
            }
        }

        return r;
    }

    size_t GetSectionCount() const throw () {
        return GetHeader()->e_shnum;
    }

    Elf64_Shdr* GetSectionBegin() const throw () {
        return reinterpret_cast<Elf64_Shdr*>(Begin + GetHeader()->e_shoff);
    }

    Elf64_Shdr* GetSectionEnd() const throw () {
        return reinterpret_cast<Elf64_Shdr*>(Begin + GetHeader()->e_shoff) + GetHeader()->e_shnum;
    }

    Elf64_Shdr* GetSection(size_t i) const throw () {
        return GetSectionBegin() + i;
    }

private:
    TFileMap Map;
    char* Begin;
};

class TSection {
public:
    TSection(TElf* elf, Elf64_Shdr* this_)
        : Elf(elf)
        , This(this_)
    {
    }

    bool IsNull() const throw () {
        return !This;
    }

    char* GetPtr(size_t offset = 0) const throw () {
        return Elf->GetPtr(This->sh_offset) + offset;
    }

    TStringBuf GetStr(size_t offset) const throw () {
        return GetPtr(offset);
    }

    TStringBuf GetName() const throw () {
        return TSection(Elf, Elf->GetSection(Elf->GetHeader()->e_shstrndx)).GetPtr(This->sh_name);
    }

    size_t GetLink() const throw () {
        return This->sh_link;
    }

    size_t GetSize() const throw () {
        return This->sh_size;
    }

    size_t GetEntryCount() const throw () {
        return GetSize() / This->sh_entsize;
    }

    template<typename TTo = char>
    TTo* GetEntry(size_t i) const throw () {
        return reinterpret_cast<TTo*>(GetPtr(i * This->sh_entsize));
    }

private:
    TElf* Elf;
    Elf64_Shdr* This;
};

class TVerneedSection : public TSection {
public:
    TVerneedSection(TElf* elf)
        : TSection(elf, elf->GetSectionByType(SHT_GNU_verneed))
    {
    }

    Elf64_Verneed* GetFirstVerneed() const throw () {
        if (!GetSize()) {
            return nullptr;
        }

        return reinterpret_cast<Elf64_Verneed*>(GetPtr());
    }

    Elf64_Verneed* GetNextVerneed(Elf64_Verneed* v) const throw () {
        if (!v->vn_next) {
            return nullptr;
        }

        return Offset<Elf64_Verneed*>(v, v->vn_next);
    }

    Elf64_Vernaux* GetFirstVernaux(Elf64_Verneed* v) const throw () {
        if (!v->vn_cnt) {
            return nullptr;
        }

        return Offset<Elf64_Vernaux*>(v, v->vn_aux);
    }

    Elf64_Vernaux* GetNextVernaux(Elf64_Vernaux* v) const throw () {
        if (!v->vna_next) {
            return nullptr;
        }

        return Offset<Elf64_Vernaux*>(v, v->vna_next);
    }
};

}
