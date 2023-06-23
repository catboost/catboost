#pragma once

#include <util/generic/buffer.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/flags.h>

class TFileEntitiesList {
public:
    enum EMaskFlag {
        EM_FILES = 1,
        EM_DIRS = 2,
        EM_SLINKS = 4,

        EM_FILES_DIRS = EM_FILES | EM_DIRS,
        EM_FILES_SLINKS = EM_FILES | EM_SLINKS,
        EM_DIRS_SLINKS = EM_DIRS | EM_SLINKS,
        EM_FILES_DIRS_SLINKS = EM_FILES | EM_DIRS | EM_SLINKS
    };
    Y_DECLARE_FLAGS(EMask, EMaskFlag);

    TFileEntitiesList(EMask mask)
        : Mask(mask)
    {
        Clear();
    }

    void Clear() {
        Cur = nullptr;
        FileNamesSize = CurName = 0;
        FileNames.Clear();
        FileNames.Append("", 1);
    }

    const char* Next() {
        return Cur = (CurName++ < FileNamesSize ? strchr(Cur, 0) + 1 : nullptr);
    }

    size_t Size() {
        return FileNamesSize;
    }

    inline void Fill(const TString& dirname, bool sort = false) {
        Fill(dirname, TStringBuf(), sort);
    }

    inline void Fill(const TString& dirname, TStringBuf prefix, bool sort = false) {
        Fill(dirname, prefix, TStringBuf(), 1, sort);
    }

    void Fill(const TString& dirname, TStringBuf prefix, TStringBuf suffix, int depth, bool sort = false);

    void Restart() {
        Cur = FileNames.Data();
        CurName = 0;
    }

protected:
    TBuffer FileNames;
    size_t FileNamesSize, CurName;
    const char* Cur;
    EMask Mask;
};

Y_DECLARE_OPERATORS_FOR_FLAGS(TFileEntitiesList::EMask);

class TFileList: public TFileEntitiesList {
public:
    TFileList()
        : TFileEntitiesList(TFileEntitiesList::EM_FILES)
    {
    }
};

class TDirsList: public TFileEntitiesList {
public:
    TDirsList()
        : TFileEntitiesList(TFileEntitiesList::EM_DIRS)
    {
    }
};
