#include "main.h"

#ifndef CATBOOST_OPENSOURCE
#include <library/cpp/charset/recyr.hh>
#endif

#include <library/cpp/containers/comptrie/comptrie.h>
#include <library/cpp/deprecated/mapped_file/mapped_file.h>
#include <library/cpp/getopt/small/last_getopt.h>

#include <util/charset/wide.h>
#include <util/generic/buffer.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/stream/buffered.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/stream/output.h>
#include <util/string/cast.h>
#include <util/string/util.h>
#include <util/system/filemap.h>

#include <string>

#ifdef WIN32
#include <crtdbg.h>
#include <windows.h>
#endif // WIN32

namespace {
    struct TOptions {
        TString Prog = {};
        TString Triefile = {};
        TString Infile = {};
        bool Minimized = false;
        bool FastLayout = false;
        TCompactTrieBuilderFlags Flags = CTBF_NONE;
        bool Unicode = false;
        bool Verify = false;
        bool Wide = false;
        bool NoValues = false;
        bool Vector = false;
        int ArraySize = 0;
        TString ValueType = {};
        bool AllowEmptyKey = false;
        bool UseAsIsParker = false;
    };
}  // namespace

static TOptions ParseOptions(const int argc, const char* argv[]) {
    TOptions options;
    auto parser = NLastGetopt::TOpts::Default();
    parser
        .AddLongOption('t', "type")
        .DefaultValue("ui64")
        .RequiredArgument("TYPE")
        .StoreResult(&options.ValueType)
        .Help("type of value or array element, possible: ui16, i16, ui32, i32, ui64, i64, float, double, bool,"
              " TString, TUtf16String (utf-8 in input)");
    parser
        .AddCharOption('0')
        .NoArgument()
        .SetFlag(&options.NoValues)
        .Help("Do not store values to produce TCompactTrieSet compatible binary");
    parser
        .AddLongOption('a', "array")
        .NoArgument()
        .SetFlag(&options.Vector)
        .Help("Values are arrays (of some type, depending on -t flag). Input looks like"
              " 'key <TAB> value1 <TAB> .. <TAB> valueN' (N may differ from line to line)");
    parser
        .AddCharOption('S')
        .RequiredArgument("INT")
        .StoreResult(&options.ArraySize)
        .Help("Values are fixed size not packed (!) arrays, may used for mapping classes/structs"
              " with monotyped fields");
    parser
        .AddLongOption('e', "allow-empty")
        .NoArgument()
        .SetFlag(&options.AllowEmptyKey)
        .Help("Allow empty key");
    parser
        .AddLongOption('i', "input")
        .DefaultValue("-")
        .RequiredArgument("FILE")
        .StoreResult(&options.Infile)
        .Help("Input file");
    parser
        .AddLongOption('m', "minimize")
        .NoArgument()
        .SetFlag(&options.Minimized)
        .Help("Minimize tree into a DAG");
    parser
        .AddLongOption('f', "fast-layout")
        .NoArgument()
        .SetFlag(&options.FastLayout)
        .Help("Make fast layout");
    parser
        .AddLongOption('v', "verbose")
        .NoArgument()
        .Help("Be verbose - show progress & stats");
    parser
        .AddLongOption('s', "prefix-grouped")
        .NoArgument()
        .Help("Assume input is prefix-grouped by key (for every prefix all keys with this prefix"
              " come in one group; greatly reduces memory usage)");
    parser
        .AddLongOption('q', "unique-keys")
        .NoArgument()
        .Help("Assume the keys are unique (will report an error otherwise)");
    parser
        .AddLongOption('c', "check")
        .NoArgument()
        .Help("Check the compiled trie (works only with an explicit input file name)");
    parser
        .AddCharOption('u')
        .NoArgument()
        .SetFlag(&options.Unicode)
        .Help("Recode keys from UTF-8 to Yandex (deprecated)");
    parser
        .AddLongOption('w', "wide")
        .NoArgument()
        .SetFlag(&options.Wide)
        .Help("Treat input keys as UTF-8, recode to TChar (wchar16)");
    parser
        .AddLongOption('P', "as-is-packer")
        .NoArgument()
        .SetFlag(&options.UseAsIsParker)
        .Help("Use AsIsParker to pack value in trie");
    parser.AddHelpOption('h');
    parser.SetFreeArgsNum(1);
    parser.SetFreeArgTitle(0, "TRIE_FILE", "Compiled trie");
    NLastGetopt::TOptsParseResult parsed{&parser, argc, argv};
    options.Triefile = parsed.GetFreeArgs().front();
    if (parsed.Has('q')) {
        options.Flags |= CTBF_UNIQUE;
    }
    if (parsed.Has('s')) {
        options.Flags |= CTBF_PREFIX_GROUPED;
    }
    if (parsed.Has('v')) {
        options.Flags |= CTBF_VERBOSE;
    }
    return options;
}

namespace {
    template <class T>
    struct TFromString {
        T operator() (const char* start, size_t len) const {
            return FromStringImpl<T>(start, len);
        }
    };

    template <>
    struct TFromString<TUtf16String> {
        TUtf16String operator ()(const char* start, size_t len) const {
            return UTF8ToWide(start, len);
        }
    };

    template <class TTKey, class TTKeyChar, class TTValue,
        class TKeyReader = TFromString<TTKey>,
        class TValueReader = TFromString<TTValue> >
    struct TRecord {
        typedef TTKey TKey;
        typedef TTKeyChar TKeyChar;
        typedef TTValue TValue;
        TKey Key;
        TValue Value;
        TString Tmp;
        bool Load(IInputStream& in, const bool allowEmptyKey, const bool noValues) {
            while (in.ReadLine(Tmp)) {
                if (!Tmp) {
                    // there is a special case for TrieSet with empty keys allowed
                    if (!(noValues && allowEmptyKey)) {
                        continue;
                    }
                }

                const size_t sep = Tmp.find('\t');
                if (sep != TString::npos) {
                    if (0 == sep && !allowEmptyKey) {
                        continue;
                    }
                    Key = TKeyReader()(Tmp.data(), sep);
                    Value = TValueReader()(Tmp.data() + sep + 1, Tmp.size() - sep - 1);
                } else if (noValues) {
                    RemoveIfLast<TString>(Tmp, '\n');
                    Key = TKeyReader()(Tmp.data(), Tmp.size());
                    Value = TValue();
                }
                return true;
            }
            return false;
        }
    };

    template <class TTKey, class TTKeyChar, class T,
        class TKeyReader = TFromString<TTKey>,
        class TValueReader = TFromString<T> >
    struct TVectorRecord {
        typedef TTKey TKey;
        typedef TTKeyChar TKeyChar;
        typedef TVector<T> TValue;
        TKey Key;
        TValue Value;
        TString Tmp;

        bool Load(IInputStream& in, const bool allowEmptyKey, const bool noValues) {
            Y_UNUSED(noValues);
            while (in.ReadLine(Tmp)) {
                if (!Tmp && !allowEmptyKey) {
                    continue;
                }

                size_t sep = Tmp.find('\t');
                if (sep == TString::npos) {
                    RemoveIfLast<TString>(Tmp, '\n');
                    Key = TKeyReader()(Tmp.data(), Tmp.size());
                    Value = TValue();
                } else {
                    Key = TKeyReader()(Tmp.data(), sep);
                    Value = TValue();
                    while (sep != Tmp.size()) {
                        size_t sep2 = Tmp.find('\t', sep + 1);
                        if (sep2 == TString::npos) {
                            sep2 = Tmp.size();
                        }

                        if (sep + 1 != sep2) {
                            Value.push_back(TValueReader()(Tmp.data() + sep + 1, sep2 - sep - 1));
                        }
                        sep = sep2;
                    }
                }
                return true;
            }
            return false;
        }
    };

    template <typename TVectorType>
    class TFixedArrayAsIsPacker {
    public:
        TFixedArrayAsIsPacker()
        : ArraySize(0)
        , SizeOfValue(0)
        {
        }
        explicit TFixedArrayAsIsPacker(size_t arraySize)
        : ArraySize(arraySize)
        , SizeOfValue(arraySize * sizeof(typename TVectorType::value_type))
        {
        }
        void UnpackLeaf(const char* p, TVectorType& t) const {
            const typename TVectorType::value_type* beg = reinterpret_cast<const typename TVectorType::value_type*>(p);
            t.assign(beg, beg + ArraySize);
        }
        void PackLeaf(char* buffer, const TVectorType& data, size_t computedSize) const {
            Y_ASSERT(computedSize == SizeOfValue && data.size() == ArraySize);
            memcpy(buffer, data.data(), computedSize);
        }
        size_t MeasureLeaf(const TVectorType& data) const {
            Y_UNUSED(data);
            Y_ASSERT(data.size() == ArraySize);
            return SizeOfValue;
        }
        size_t SkipLeaf(const char* ) const {
            return SizeOfValue;
        }
    private:
        size_t ArraySize;
        size_t SizeOfValue;
    };

#ifndef CATBOOST_OPENSOURCE
    struct TUTF8ToYandexRecoder {
    TString operator()(const char* s, size_t len) {
            return Recode(CODES_UTF8, CODES_YANDEX, TString(s, len));
        }
    };
#endif

    struct TUTF8ToWideRecoder {
        TUtf16String operator()(const char* s, size_t len) {
            return UTF8ToWide(s, len);
        }
    };
}  // namespace

template <class TRecord, class TPacker>
static int ProcessFile(IInputStream& in, const TOptions& o, const TPacker& packer) {
    TFixedBufferFileOutput out(o.Triefile);
    typedef typename TRecord::TKeyChar TKeyChar;
    typedef typename TRecord::TValue TValue;

    THolder< TCompactTrieBuilder<TKeyChar, TValue, TPacker> > builder(new TCompactTrieBuilder<TKeyChar, TValue, TPacker>(o.Flags, packer));

    TRecord r;
    while (r.Load(in, o.AllowEmptyKey, o.NoValues)) {
        builder->Add(r.Key.data(), r.Key.size(), r.Value);
    }

    if (o.Flags & CTBF_VERBOSE) {
        Cerr << Endl;
        Cerr << "Entries: " << builder->GetEntryCount() << Endl;
        Cerr << "Tree nodes: " << builder->GetNodeCount() << Endl;
    }
    TBufferOutput inputForFastLayout;
    IOutputStream* currentOutput = &out;
    if (o.FastLayout) {
        currentOutput = &inputForFastLayout;
    }
    if (o.Minimized) {
        TBufferOutput raw;
        size_t datalength = builder->Save(raw);
        if (o.Flags & CTBF_VERBOSE)
            Cerr << "Data length (before compression): " << datalength << Endl;
        builder.Destroy();

        datalength = CompactTrieMinimize(*currentOutput, raw.Buffer().Data(), raw.Buffer().Size(), o.Flags & CTBF_VERBOSE, packer);
        if (o.Flags & CTBF_VERBOSE)
            Cerr << "Data length (minimized): " << datalength << Endl;
    } else {
        size_t datalength = builder->Save(*currentOutput);
        if (o.Flags & CTBF_VERBOSE)
            Cerr << "Data length: " << datalength << Endl;
    }
    if (o.FastLayout) {
        builder.Destroy();
        size_t datalength = CompactTrieMakeFastLayout(out, inputForFastLayout.Buffer().Data(),
            inputForFastLayout.Buffer().Size(), o.Flags & CTBF_VERBOSE, packer);
        if (o.Flags & CTBF_VERBOSE)
            Cerr << "Data length (fast layout): " << datalength << Endl;
    }

    return 0;
}

template <class TRecord, class TPacker>
static int VerifyFile(const TOptions& o, const TPacker& packer) {
    TMappedFile filemap(o.Triefile);
    typedef typename TRecord::TKeyChar TKeyChar;
    typedef typename TRecord::TValue TValue;
    TCompactTrie<TKeyChar, TValue, TPacker> trie((const char*)filemap.getData(), filemap.getSize(), packer);

    TFileInput in(o.Infile);
    size_t entrycount = 0;
    int retcode = 0;
    TRecord r;
    while (r.Load(in, o.AllowEmptyKey, o.NoValues)) {
        entrycount++;
        TValue trievalue;

        if (!trie.Find(r.Key.data(), r.Key.size(), &trievalue)) {
            Cerr << "Trie check failed on key #" << entrycount << "\"" << r.Key << "\": no key present" << Endl;
            retcode = 1;
        } else if (!o.NoValues && trievalue != r.Value) {
            Cerr << "Trie check failed on key #" << entrycount << "\"" << r.Key << "\": value mismatch" << Endl;
            retcode = 1;
        }
    }

    for (typename TCompactTrie<TKeyChar, TValue, TPacker>::TConstIterator iter = trie.Begin(); iter != trie.End(); ++iter) {
        entrycount--;
    }

    if (entrycount) {
        Cerr << "Broken iteration: entry count mismatch" << Endl;
        retcode = 1;
    }

    if ((o.Flags & CTBF_VERBOSE) && !retcode) {
        Cerr << "Trie check successful" << Endl;
    }
    return retcode;
}

template <class TRecord, class TPacker>
static int SelectInput(const TOptions& o, const TPacker& packer) {
    if ("-"sv == o.Infile) {
        TBufferedInput wrapper{&Cin};
        return ProcessFile<TRecord>(wrapper, o, packer);
    }

    TFileInput in(o.Infile);
    return ProcessFile<TRecord>(in, o, packer);
}

template <class TRecord, class TPacker>
static int DoMain(const TOptions& o, const TPacker& packer) {
    int retcode = SelectInput<TRecord>(o, packer);
    if (!retcode && o.Verify && !o.Triefile.empty())
        retcode = VerifyFile<TRecord>(o, packer);
    return retcode;
}

// TRecord - nested template parameter
template<class TValue,
    template<class TKey, class TKeyChar, class TValueOther,
        class TKeyReader,
        class TValueReader> class TRecord, class TPacker>
static int ProcessInput(const TOptions& o, const TPacker& packer) {
   if (!o.Wide) {
        if (!o.Unicode) {
            return DoMain< TRecord< TString, char, TValue, TFromString<TString>, TFromString<TValue> > >(o, packer);
        } else {
        #ifndef CATBOOST_OPENSOURCE
            return DoMain< TRecord< TString, char, TValue, TUTF8ToYandexRecoder, TFromString<TValue> > >(o, packer);
        #else
            Y_ABORT("Yandex encoding is not supported in CATBOOST_OPENSOURCE mode");
        #endif
        }
    } else {
        return DoMain< TRecord< TUtf16String, TChar, TValue, TUTF8ToWideRecoder, TFromString<TValue> > >(o, packer);
    }
 }

template <class TItemType>
static int ProcessInput(const TOptions& o) {
    if (o.ArraySize > 0) {
        return ProcessInput<TItemType, TVectorRecord>(o, TFixedArrayAsIsPacker<TVector<TItemType> >(o.ArraySize));
    } else if (o.Vector) {
        return ProcessInput<TItemType, TVectorRecord>(o, TCompactTriePacker<TVector<TItemType> >());
    } else if (o.UseAsIsParker) {
        return ProcessInput<TItemType, TRecord>(o, TAsIsPacker<TItemType>());
    } else {
        return ProcessInput<TItemType, TRecord>(o, TCompactTriePacker<TItemType>());
    }
}

static int Main(const int argc, const char* argv[])
try {
#ifdef WIN32
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    ::SetConsoleCP(1251);
    ::SetConsoleOutputCP(1251);
#endif // WIN32
    const TOptions o = ParseOptions(argc, argv);
    if (o.NoValues) {
        return ProcessInput<ui64, TRecord>(o, TNullPacker<ui64>());
    } else {
#define CHECK_TYPE_AND_PROCESS(valueType)          \
            if (o.ValueType == #valueType) {       \
                return ProcessInput<valueType>(o); \
            }
        CHECK_TYPE_AND_PROCESS(ui16)
        CHECK_TYPE_AND_PROCESS(i16)
        CHECK_TYPE_AND_PROCESS(ui32)
        CHECK_TYPE_AND_PROCESS(i32)
        CHECK_TYPE_AND_PROCESS(ui64)
        CHECK_TYPE_AND_PROCESS(i64)
        CHECK_TYPE_AND_PROCESS(bool)
        CHECK_TYPE_AND_PROCESS(float)
        CHECK_TYPE_AND_PROCESS(double)
        CHECK_TYPE_AND_PROCESS(TString)
        CHECK_TYPE_AND_PROCESS(TUtf16String)
#undef CHECK_TYPE_AND_PROCESS
        ythrow yexception() << "unknown type for -t option: " << o.ValueType;
    }
} catch (const std::exception& e) {
    Cerr << "Exception: " << e.what() << Endl;
    return 2;
} catch (...) {
    Cerr << "Unknown exception!\n";
    return 3;
}

int NTrieOps::MainCompile(const int argc, const char* argv[]) {
    return ::Main(argc, argv);
}
