#include "mapped_file.h"

#include <util/generic/yexception.h>
#include <util/system/defaults.h>
#include <util/system/hi_lo.h>
#include <util/system/filemap.h>

TMappedFile::TMappedFile(TFileMap* map, const char* dbgName) {
    Map_ = map;
    i64 len = Map_->Length();
    if (Hi32(len) != 0 && sizeof(size_t) <= sizeof(ui32))
        ythrow yexception() << "File '" << dbgName << "' mapping error: " << len << " too large";

    Map_->Map(0, static_cast<size_t>(len));
}

TMappedFile::TMappedFile(const TFile& file, TFileMap::EOpenMode om, const char* dbgName)
    : Map_(nullptr)
{
    init(file, om, dbgName);
}

void TMappedFile::precharge(size_t off, size_t size) const {
    if (!Map_)
        return;

    Map_->Precharge(off, size);
}

void TMappedFile::init(const TString& name) {
    THolder<TFileMap> map(new TFileMap(name));
    TMappedFile newFile(map.Get(), name.data());
    Y_UNUSED(map.Release());
    newFile.swap(*this);
    newFile.term();
}

void TMappedFile::init(const TString& name, size_t length, TFileMap::EOpenMode om) {
    THolder<TFileMap> map(new TFileMap(name, length, om));
    TMappedFile newFile(map.Get(), name.data());
    Y_UNUSED(map.Release());
    newFile.swap(*this);
    newFile.term();
}

void TMappedFile::init(const TFile& file, TFileMap::EOpenMode om, const char* dbgName) {
    THolder<TFileMap> map(new TFileMap(file, om));
    TMappedFile newFile(map.Get(), dbgName);
    Y_UNUSED(map.Release());
    newFile.swap(*this);
    newFile.term();
}

void TMappedFile::init(const TString& name, TFileMap::EOpenMode om) {
    THolder<TFileMap> map(new TFileMap(name, om));
    TMappedFile newFile(map.Get(), name.data());
    Y_UNUSED(map.Release());
    newFile.swap(*this);
    newFile.term();
}

void TMappedFile::flush() {
    Map_->Flush();
}
