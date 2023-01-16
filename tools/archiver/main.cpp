#include <library/cpp/archive/yarchive.h>
#include <library/cpp/deprecated/mapped_file/mapped_file.h>
#include <library/cpp/digest/md5/md5.h>
#include <library/cpp/getopt/small/last_getopt.h>

#include <util/folder/dirut.h>
#include <util/folder/filelist.h>
#include <util/folder/path.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/memory/blob.h>
#include <util/stream/file.h>
#include <util/string/cast.h>
#include <util/string/escape.h>
#include <util/string/hex.h>
#include <util/string/subst.h>
#include <util/system/filemap.h>

#include <cstring>

namespace {
    class TStringArrayOutput: public IOutputStream {
    public:
        TStringArrayOutput(IOutputStream* slave, size_t stride)
            : Slave(*slave)
            , Stride(stride)
        {
            Buf.reserve(stride);
        }
        void DoFinish() override {
            WriteBuf();
            Flush();
        }
        void DoWrite(const void* data, size_t len) override {
            for (const char* p = (const char*)data; len > 0; ++p, --len) {
                Buf.append(*p);
                if (Buf.size() == Stride)
                    WriteBuf();
            }
        }

    private:
        void WriteBuf() {
            Slave << '"' << Buf << "\",\n"sv;
            Buf.clear();
        }

    private:
        IOutputStream& Slave;
        const size_t Stride;
        TString Buf;
    };

    class THexOutput: public IOutputStream {
    public:
        inline THexOutput(IOutputStream* slave)
            : Slave_(slave)
        {
        }

        ~THexOutput() override {
        }

        inline IOutputStream* Slave() const noexcept {
            return Slave_;
        }

    private:
        void DoFinish() override {
            Slave_->Write('\n');
            Slave_->Flush();
        }

        void DoWrite(const void* data, size_t len) override {
            const char* b = (const char*)data;

            while (len) {
                const unsigned char c = *b;
                char buf[12];
                char* tmp = buf;

                if (Count_ % Columns == 0) {
                    *tmp++ = ' ';
                    *tmp++ = ' ';
                    *tmp++ = ' ';
                    *tmp++ = ' ';
                }

                if (Count_ && Count_ % Columns != 0) {
                    *tmp++ = ',';
                    *tmp++ = ' ';
                }

                *tmp++ = '0';
                *tmp++ = 'x';
                tmp = HexEncode(&c, 1, tmp);

                if ((Count_ % Columns) == (Columns - 1)) {
                    *tmp++ = ',';
                    *tmp++ = '\n';
                }

                Slave_->Write(buf, tmp - buf);

                --len;
                ++b;
                ++Count_;
            }
        }

    private:
        // width in source chars
        static const size_t Columns = 10;
        ui64 Count_ = 0;
        IOutputStream* Slave_ = nullptr;
    };

    struct TYasmOutput: public IOutputStream {
        inline TYasmOutput(IOutputStream* out, const TString& base)
            : Out_(out)
            , Base_(base)
        {
            *Out_ << "global " << Base_ << "\n";
            *Out_ << "global " << Base_ << "Size\n\nSECTION .rodata\n\n";
            *Out_ << Base_ << ":\n";
        }

        ~TYasmOutput() override {
        }

        void DoFinish() override {
            *Out_ << Base_ << "Size:\ndd " << Count_ << '\n';

            *Out_ << "%ifidn __OUTPUT_FORMAT__,elf64\n";
            *Out_ << "size " << Base_ << " " << Count_ << "\n";
            *Out_ << "size " << Base_ << "Size 4\n";
            *Out_ << "%endif\n";
        }

        void DoWrite(const void* data, size_t len) override {
            Count_ += len;

            const unsigned char* p = (const unsigned char*)data;

            while (len) {
                const size_t step = Min<size_t>(len, 100);

                *Out_ << "db " << (int)*p++;

                for (size_t i = 1; i < step; ++i) {
                    *Out_ << ',' << (int)*p++;
                }

                *Out_ << '\n';

                len -= step;
            }
        }

        IOutputStream* Out_ = nullptr;
        const TString Base_;
        ui64 Count_ = 0;
    };

    struct TCOutput: public THexOutput {
        inline TCOutput(IOutputStream* out, const TString& base)
            : THexOutput(out)
            , B(base)
        {
            *Slave() << "static_assert(sizeof(unsigned int) == 4, \"ups, unsupported platform\");\n\nextern \"C\" {\nextern const unsigned char " << B << "[] = {\n";
        }

        ~TCOutput() override {
        }

        void DoFinish() override {
            *Slave() << "\n};\nextern const unsigned int " << B << "Size = sizeof(" << B << ") / sizeof(" << B << "[0]);\n}\n";
        }

        const TString B;
    };

    struct TCStringOutput: public IOutputStream {
        inline TCStringOutput(IOutputStream* out, const TString& base)
            : O(out)
            , B(base)
        {
            *O << "static_assert(sizeof(unsigned int) == 4, \"ups, unsupported platform\");\n\nextern \"C\" {\nextern const unsigned char " << B << "[] = \n";
        }

        ~TCStringOutput() override {
        }

        void DoWrite(const void* data, size_t len) override {
            *O << TString((const char*)data, len).Quote() << '\n';
        }

        void DoFinish() override {
            //*O << ";\nextern const unsigned char* " << B << " = (const unsigned char*)" << B << "Array;\n";
            *O << ";\nextern const unsigned int " << B << "Size = sizeof(" << B << ") / sizeof(" << B << "[0]) - 1;\n}\n";
        }

        IOutputStream* O = nullptr;
        const TString B;
    };

    struct TMyFileComparator {
        bool operator()(const TString& fname1, const TString& fname2) const {
            if (fname1 == fname2) {
                return false;
            }
            if (const auto* savedResultPtr = SavedResults.FindPtr(std::make_pair(fname1, fname2))) {
                return *savedResultPtr < 0;
            }
            TMemoryMap mmap1(fname1, TMemoryMap::oRdOnly);
            TMemoryMap mmap2(fname2, TMemoryMap::oRdOnly);
            mmap1.SetSequential();
            mmap2.SetSequential();
            Y_ASSERT(mmap1.Length() == mmap2.Length());
            TMemoryMap::TMapResult mapResult1 = mmap1.Map(0, mmap1.Length());
            TMemoryMap::TMapResult mapResult2 = mmap2.Map(0, mmap2.Length());
            Y_ASSERT(mapResult1.MappedSize() == mapResult2.MappedSize());
            int res = memcmp(mapResult1.MappedData(), mapResult2.MappedData(), mapResult1.MappedSize());
            mmap1.Unmap(mapResult1);
            mmap2.Unmap(mapResult2);
            SavedResults[std::make_pair(fname1, fname2)] = res;
            SavedResults[std::make_pair(fname2, fname1)] = -res;
            return res < 0;
        }

        mutable THashMap<std::pair<TString, TString>, int> SavedResults;
    };

    struct TDuplicatesMap {
        void Add(const TString& fname, const TString& rname) {
            Y_ENSURE(!InitialFillingDone);
            FileNames.push_back(fname);
            FileNameToRecordName[fname] = rname;
        }

        void Finish() {
            Y_ENSURE(!InitialFillingDone);
            InitialFillingDone = true;
            TMap<i64, TVector<TString>> bySize;
            for (const TString& fname: FileNames) {
                TFile file(fname, OpenExisting | RdOnly);
                bySize[file.GetLength()].push_back(fname);
            }
            for (const auto& bySizeElement: bySize) {
                if (bySizeElement.second.size() > 1) {
                    TMap<TString, TVector<TString>, TMyFileComparator> byContents;
                    for (const TString& fname: bySizeElement.second) {
                        byContents[fname].push_back(fname);
                    }
                    for (const auto& byContentsElement: byContents) {
                        if (byContentsElement.second.size() > 1) {
                            const TString& rootName = byContentsElement.second.front();
                            const TString& rootRecordName = FileNameToRecordName[rootName];
                            for (const TString& fname: byContentsElement.second) {
                                if (fname != rootName) {
                                    Synonyms[FileNameToRecordName[fname]] = rootRecordName;
                                }
                            }
                        }
                    }
                }
            }
            FileNames.clear();
            FileNameToRecordName.clear();
        }

        bool InitialFillingDone = false;
        TVector<TString> FileNames;
        THashMap<TString, TString> FileNameToRecordName;
        THashMap<TString, TString> Synonyms;
    };

    struct TDeduplicationArchiveWriter {
        TDeduplicationArchiveWriter(const TDuplicatesMap& duplicatesMap, IOutputStream* out, bool compress)
            : DuplicatesMap(duplicatesMap)
            , Writer(out, compress)
        {}

        void Finish() {
            Writer.Finish();
        }

        const TDuplicatesMap& DuplicatesMap;
        TArchiveWriter Writer;
    };
}

static inline TAutoPtr<IOutputStream> OpenOutput(const TString& url) {
    if (url.empty()) {
        return new TBuffered<TUnbufferedFileOutput>(8192, Duplicate(1));
    } else {
        return new TBuffered<TUnbufferedFileOutput>(8192, url);
    }
}

static inline bool IsDelim(char ch) noexcept {
    return ch == '/' || ch == '\\';
}

static inline TString GetFile(const TString& s) {
    const char* e = s.end();
    const char* b = s.begin();
    const char* c = e - 1;

    while (c != b && !IsDelim(*c)) {
        --c;
    }

    if (c != e && IsDelim(*c)) {
        ++c;
    }

    return TString(c, e - c);
}

static inline TString Fix(TString f) {
    if (!f.empty() && IsDelim(f[f.size() - 1])) {
        f.pop_back();
    }

    return f;
}

static bool Quiet = false;

static inline void Append(IOutputStream& w, const TString& fname, const TString& rname) {
    TMappedFileInput in(fname);

    if (!Quiet) {
        Cerr << "--> " << rname << Endl;
    }

    TransferData((IInputStream*)&in, &w);
}

static inline void Append(TDuplicatesMap& w, const TString& fname, const TString& rname) {
    w.Add(fname, rname);
}

static inline void Append(TDeduplicationArchiveWriter& w, const TString& fname, const TString& rname) {
    if (!Quiet) {
        Cerr << "--> " << rname << Endl;
    }

    if (const TString* rootRecordName = w.DuplicatesMap.Synonyms.FindPtr(rname)) {
        w.Writer.AddSynonym(*rootRecordName, rname);
    } else {
        TMappedFileInput in(fname);
        w.Writer.Add(rname, &in);
    }
}

namespace {
    struct TRec {
        bool Recursive = false;
        TString Key;
        TString Path;
        TString Prefix;

        TRec() = default;

        inline void Fix() {
            ::Fix(Path);
            ::Fix(Prefix);
        }

        template <typename T>
        inline void Recurse(T& w) const {
            if (IsDir(Path)) {
                DoRecurse(w, "/");
            } else {
                Append(w, Path, Key.size() ? Key : Prefix + "/" + GetFile(Path));
            }
        }

        template <typename T>
        inline void DoRecurse(T& w, const TString& off) const {
            {
                TFileList fl;

                const char* name;
                const TString p = Path + off;

                fl.Fill(p, true);

                while ((name = fl.Next())) {
                    const TString fname = p + name;
                    const TString rname = Prefix + off + name;

                    Append(w, fname, rname);
                }
            }

            if (Recursive) {
                TDirsList dl;

                const char* name;
                const TString p = Path + off;

                dl.Fill(p, true);

                while ((name = dl.Next())) {
                    if (strcmp(name, ".") && strcmp(name, "..")) {
                        DoRecurse(w, off + name + "/");
                    }
                }
            }
        }
    };
}

static TString CutFirstSlash(const TString& fileName) {
    if (fileName[0] == '/') {
        return fileName.substr(1);
    } else {
        return fileName;
    }
}

struct TMappingReader {
    TMemoryMap Map;
    TBlob Blob;
    TArchiveReader Reader;

    TMappingReader(const TString& archive)
        : Map(archive)
        , Blob(TBlob::FromMemoryMapSingleThreaded(Map, 0, Map.Length()))
        , Reader(Blob)
    {
    }
};

static void UnpackArchive(const TString& archive, const TFsPath& dir = TFsPath()) {
    TMappingReader mappingReader(archive);
    const TArchiveReader& reader = mappingReader.Reader;
    const size_t count = reader.Count();
    for (size_t i = 0; i < count; ++i) {
        const TString key = reader.KeyByIndex(i);
        const TString fileName = CutFirstSlash(key);
        if (!Quiet) {
            Cerr << archive << " --> " << fileName << Endl;
        }
        const TFsPath path(dir / fileName);
        path.Parent().MkDirs();
        TAutoPtr<IInputStream> in = reader.ObjectByKey(key);
        TFixedBufferFileOutput out(path);
        TransferData(in.Get(), &out);
        out.Finish();
    }
}

static void ListArchive(const TString& archive, bool cutSlash) {
    TMappingReader mappingReader(archive);
    const TArchiveReader& reader = mappingReader.Reader;
    const size_t count = reader.Count();
    for (size_t i = 0; i < count; ++i) {
        const TString key = reader.KeyByIndex(i);
        TString fileName = key;
        if (cutSlash) {
            fileName = CutFirstSlash(key);
        }
        Cout << fileName << Endl;
    }
}

static void ListArchiveMd5(const TString& archive, bool cutSlash) {
    TMappingReader mappingReader(archive);
    const TArchiveReader& reader = mappingReader.Reader;
    const size_t count = reader.Count();
    for (size_t i = 0; i < count; ++i) {
        const TString key = reader.KeyByIndex(i);
        TString fileName = key;
        if (cutSlash) {
            fileName = CutFirstSlash(key);
        }
        char md5buf[33];
        Cout << fileName << '\t' << MD5::Stream(reader.ObjectByKey(key).Get(), md5buf) << Endl;
    }
}

int main(int argc, char** argv) {
    NLastGetopt::TOpts opts;
    opts.AddHelpOption('?');
    opts.SetTitle(
        "Archiver\n"
        "Docs: https://wiki.yandex-team.ru/Development/Poisk/arcadia/tools/archiver"
    );

    bool hexdump = false;
    opts.AddLongOption('x', "hexdump", "Produce hexdump")
        .NoArgument()
        .Optional()
        .StoreValue(&hexdump, true);

    size_t stride = 0;
    opts.AddLongOption('s', "segments", "Produce segmented C strings array of given size")
        .RequiredArgument("<size>")
        .Optional()
        .DefaultValue("0")
        .StoreResult(&stride);

    bool cat = false;
    opts.AddLongOption('c', "cat", "Do not store keys (file names), just cat uncompressed files")
        .NoArgument()
        .Optional()
        .StoreValue(&cat, true);

    bool doNotZip = false;
    opts.AddLongOption('p', "plain", "Do not use compression")
        .NoArgument()
        .Optional()
        .StoreValue(&doNotZip, true);

    bool deduplicate = false;
    opts.AddLongOption("deduplicate", "Turn on file-wise deduplication")
        .NoArgument()
        .Optional()
        .StoreValue(&deduplicate, true);

    bool unpack = false;
    opts.AddLongOption('u', "unpack", "Unpack archive into current directory")
        .NoArgument()
        .Optional()
        .StoreValue(&unpack, true);

    bool list = false;
    opts.AddLongOption('l', "list", "List files in archive")
        .NoArgument()
        .Optional()
        .StoreValue(&list, true);

    bool cutSlash = true;
    opts.AddLongOption("as-is", "somewhy slash is cutted by default in list; with this option key will be shown as-is")
        .NoArgument()
        .Optional()
        .StoreValue(&cutSlash, false);

    bool listMd5 = false;
    opts.AddLongOption('m', "md5", "List files in archive with MD5 sums")
        .NoArgument()
        .Optional()
        .StoreValue(&listMd5, true);

    bool recursive = false;
    opts.AddLongOption('r', "recursive", "Read all files under each directory, recursively")
        .NoArgument()
        .Optional()
        .StoreValue(&recursive, true);

    Quiet = false;
    opts.AddLongOption('q', "quiet", "Do not output progress to stderr")
        .NoArgument()
        .Optional()
        .StoreValue(&Quiet, true);

    TString prepend;
    opts.AddLongOption('z', "prepend", "Prepend string to output")
        .RequiredArgument("<prefix>")
        .StoreResult(&prepend);

    TString append;
    opts.AddLongOption('a', "append", "Append string to output")
        .RequiredArgument("<suffix>")
        .StoreResult(&append);

    TString outputf;
    opts.AddLongOption('o', "output", "Output to file instead stdout")
        .RequiredArgument("<file>")
        .StoreResult(&outputf);

    TString unpackDir;
    opts.AddLongOption('d', "unpackdir", "Unpack destination directory")
        .RequiredArgument("<dir>")
        .DefaultValue(".")
        .StoreResult(&unpackDir);

    TString yasmBase;
    opts.AddLongOption('A', "yasm", "Output dump is yasm format")
        .RequiredArgument("<base>")
        .StoreResult(&yasmBase);

    TString cppBase;
    opts.AddLongOption('C', "cpp", "Output dump is C/C++ format")
        .RequiredArgument("<base>")
        .StoreResult(&cppBase);

    TString forceKeys;
    opts.AddLongOption('k', "keys", "Set explicit list of keys for elements")
        .RequiredArgument("<keys>")
        .StoreResult(&forceKeys);

    opts.SetFreeArgDefaultTitle("<file>");
    opts.SetFreeArgsMin(1);
    NLastGetopt::TOptsParseResult optsRes(&opts, argc, argv);

    SubstGlobal(append, "\\n", "\n");
    SubstGlobal(prepend, "\\n", "\n");

    TVector<TRec> recs;
    const auto& files = optsRes.GetFreeArgs();

    TVector<TStringBuf> keys;
    if (forceKeys.size())
        StringSplitter(forceKeys).Split(':').SkipEmpty().Collect(&keys);

    if (keys.size() && keys.size() != files.size()) {
        Cerr << "Invalid number of keys=" << keys.size() << " (!= number of files=" << files.size() << ")" << Endl;
        return 1;
    }

    for (size_t i = 0; i < files.size(); ++i) {
        const auto& path = files[i];
        size_t off = 0;
#ifdef _win_
        if (path[0] > 0 && isalpha(path[0]) && path[1] == ':')
            off = 2; // skip drive letter ("d:")
#endif               // _win_
        const size_t pos = path.find(':', off);
        TRec cur;
        cur.Path = path.substr(0, pos);
        if (pos != TString::npos)
            cur.Prefix = path.substr(pos + 1);
        if (keys.size())
            cur.Key = keys[i];
        cur.Recursive = recursive;
        cur.Fix();
        recs.push_back(cur);
    }

    try {
        if (listMd5) {
            for (const auto& rec: recs) {
                ListArchiveMd5(rec.Path, cutSlash);
            }
        } else if (list) {
            for (const auto& rec: recs) {
                ListArchive(rec.Path, cutSlash);
            }
        } else if (unpack) {
            const TFsPath dir(unpackDir);
            for (const auto& rec: recs) {
                UnpackArchive(rec.Path, dir);
            }
        } else {
            TAutoPtr<IOutputStream> outf(OpenOutput(outputf));
            IOutputStream* out = outf.Get();
            THolder<IOutputStream> hexout;

            if (hexdump) {
                hexout.Reset(new THexOutput(out));
                out = hexout.Get();
            } else if (stride) {
                hexout.Reset(new TStringArrayOutput(out, stride));
                out = hexout.Get();
            } else if (yasmBase) {
                hexout.Reset(new TYasmOutput(out, yasmBase));
                out = hexout.Get();
            } else if (cppBase) {
                hexout.Reset(new TCStringOutput(out, cppBase));
                out = hexout.Get();
            }

            outf->Write(prepend.data(), prepend.size());

            if (cat) {
                for (const auto& rec: recs) {
                    rec.Recurse(*out);
                }
            } else {
                TDuplicatesMap duplicatesMap;
                if (deduplicate) {
                    for (const auto& rec: recs) {
                        rec.Recurse(duplicatesMap);
                    }
                }
                duplicatesMap.Finish();
                TDeduplicationArchiveWriter w(duplicatesMap, out, !doNotZip);
                for (const auto& rec: recs) {
                    rec.Recurse(w);
                }
                w.Finish();
            }

            try {
                out->Finish();
            } catch (...) {
            }

            outf->Write(append.data(), append.size());
        }
    } catch (...) {
        Cerr << CurrentExceptionMessage() << Endl;
        return 1;
    }

    return 0;
}
