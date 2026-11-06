// pyres_pack: compile a group of python sources of one module and pack them into a single "pyres pack".
//
// For every (--srcx ROOTREL- --dest py/mod/name.py --src path/to/name.py --srcpath SRCPATH) set it emits
// the same resfs keys as PY_SRCS + RESOURCE_FILES used to:
//
//   resfs/src/resfs/file/<dest>          = SRCPATH              (with --no-py omitted)
//   resfs/file/<dest>                    = <source>             (with --no-py omitted)
//   resfs/src/resfs/file/<dest>.yapyc3   = SRCPATH.yapyc3       (with --no-pyc omitted)
//   resfs/file/<dest>.yapyc3             = marshal(compile())   (with --no-pyc omitted)
//
// SRCPATH is the source path relative to the source or the build root as ymake renders it (${rootrel}),
// ROOTREL is the file name compiled into the bytecode, as tools/py3cc used to get it.
//
// plus a record per `--kv=KEY=VALUE` (resources PY_SRCS used to register with RESOURCE, e.g. py/namespace).
//
// Everything after --output normally comes in a ya command file (--ya-start-command-file): an
// `@path` argument expands to the lines of that file, one argument per line.
//
// Values are compressed with NResource::Compress (the codec library/cpp/resource decompresses with).
// The pack is a plain file (a global output of the module, see _PY3_RESFS_PACK): it never reaches the
// linker, library/python/runtime_py3/pyres_ld_plugin.py picks the packs up at link time.
//
// Pack layout (little-endian, 8-byte aligned):
//   char[8] magic "YPYPACK1"; u64 size (of the whole pack); u32 count; u32 flags;
//   records: u32 klen, u32 vlen, key, value, zero padding to 8 bytes.

#include <library/cpp/archive/yarchive.h>
#include <library/cpp/resource/registry.h>

#include <util/generic/algorithm.h>
#include <util/generic/maybe.h>
#include <util/generic/scope.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/memory/blob.h>
#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/string/ascii.h>

#include <Python.h>
#include <marshal.h>

#include <algorithm>
#include <cstring>

namespace {
    struct TPyObjDeleter {
        static void Destroy(PyObject* o) noexcept {
            Py_XDECREF(o);
        }
    };
    using TPyObject = THolder<PyObject, TPyObjDeleter>;

    constexpr TStringBuf PACK_MAGIC = "YPYPACK1";

    struct TArgs {
        TString Output;
        bool NoPy = false;
        bool NoPyc = false;
        TVector<TString> SrcX;
        TVector<TString> SrcPath;
        TVector<TString> Dest;
        TVector<TString> Src;
        TVector<TString> Kv;
    };

    [[noreturn]] void Usage(const char* argv0) {
        Cerr << "Usage:\n\t" << argv0
             << " --output OUT.pyres [--no-py] [--no-pyc]"
                " [--srcx ROOTREL-... --dest DEST... --src SRC... --srcpath SRCPATH...] [--kv=KEY=VALUE...]"
             << Endl;
        exit(1);
    }

    // ya replaces --ya-start-command-file ... --ya-end-command-file with `@file`; executors that do
    // not support command files just drop the markers, so both forms are accepted.
    TVector<TString> ExpandArgs(int argc, char** argv) {
        TVector<TString> argList;
        for (int i = 1; i < argc; ++i) {
            const TStringBuf arg(argv[i]);
            if (arg == "--ya-start-command-file" || arg == "--ya-end-command-file") {
                continue;
            }
            if (!arg.StartsWith('@')) {
                argList.emplace_back(arg);
                continue;
            }
            TFileInput in(TString(arg.substr(1)));
            TString line;
            while (in.ReadLine(line)) {
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (!line.empty()) {
                    argList.push_back(line);
                }
            }
        }
        return argList;
    }

    TArgs ParseArgs(const char* argv0, const TVector<TString>& argList) {
        TArgs args;
        const size_t argc = argList.size();
        for (size_t i = 0; i < argc; ++i) {
            const TStringBuf opt(argList[i]);
            auto value = [&]() -> TString {
                if (i + 1 >= argc) {
                    Cerr << "missing value for " << opt << Endl;
                    Usage(argv0);
                }
                return argList[++i];
            };
            // list options accept either one value per occurrence or all values up to the next option
            auto values = [&](TVector<TString>& into) {
                into.push_back(value());
                while (i + 1 < argc && !argList[i + 1].StartsWith("--")) {
                    into.push_back(argList[++i]);
                }
            };
            if (opt == "--output") {
                args.Output = value();
            } else if (opt == "--no-py") {
                args.NoPy = true;
            } else if (opt == "--no-pyc") {
                args.NoPyc = true;
            } else if (opt == "--srcx") {
                values(args.SrcX);
            } else if (opt == "--srcpath") {
                values(args.SrcPath);
            } else if (opt == "--dest") {
                values(args.Dest);
            } else if (opt == "--src") {
                values(args.Src);
            } else if (opt.StartsWith("--kv=")) {
                args.Kv.emplace_back(opt.substr(5));
            } else {
                Cerr << "unknown option " << opt << Endl;
                Usage(argv0);
            }
        }
        if (args.Output.empty() || (args.Src.empty() && args.Kv.empty()) ||
            args.Src.size() != args.Dest.size() || args.Src.size() != args.SrcX.size() || args.Src.size() != args.SrcPath.size()) {
            Usage(argv0);
        }
        for (const TString& kv : args.Kv) {
            if (kv.find('=') == TString::npos) {
                Cerr << "--kv= value must be KEY=VALUE: " << kv << Endl;
                Usage(argv0);
            }
        }
        return args;
    }

    // --- bytecode ------------------------------------------------------------------------------

    // Strict UTF-8 as the CPython decoder accepts it: no overlong forms, surrogates or code points
    // above U+10FFFF.
    bool IsValidUtf8(TStringBuf s) {
        const auto* p = reinterpret_cast<const unsigned char*>(s.data());
        const auto* end = p + s.size();
        while (p < end) {
            const unsigned char c = *p++;
            if (c < 0x80) {
                continue;
            }
            size_t n;
            ui32 cp;
            if (c >= 0xC2 && c <= 0xDF) {
                n = 1, cp = c & 0x1F;
            } else if (c >= 0xE0 && c <= 0xEF) {
                n = 2, cp = c & 0x0F;
            } else if (c >= 0xF0 && c <= 0xF4) {
                n = 3, cp = c & 0x07;
            } else {
                return false;
            }
            if (static_cast<size_t>(end - p) < n) {
                return false;
            }
            for (size_t i = 0; i < n; ++i, ++p) {
                if ((*p & 0xC0) != 0x80) {
                    return false;
                }
                cp = (cp << 6) | (*p & 0x3F);
            }
            if ((n == 2 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) || (n == 3 && (cp < 0x10000 || cp > 0x10FFFF))) {
                return false;
            }
        }
        return true;
    }

    enum class ECookie {
        None,
        Utf8,
        Latin1,
        Other,
    };

    // The PEP 263 coding cookie of the source, classified the way the CPython tokenizer does
    // (check_coding_spec, get_normal_name): on the first line, or on the second one if the first
    // line is blank or a comment.
    ECookie CodingCookie(TStringBuf source) {
        auto spec = [](TStringBuf line, bool& comment) -> TMaybe<TString> {
            size_t i = 0;
            while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\014')) {
                ++i;
            }
            comment = i == line.size() || line[i] == '#' || line[i] == '\r';
            if (i == line.size() || line[i] != '#') {
                return Nothing();
            }
            for (size_t pos = line.find("coding", i); pos != TStringBuf::npos; pos = line.find("coding", pos + 1)) {
                size_t j = pos + 6;
                if (j >= line.size() || (line[j] != ':' && line[j] != '=')) {
                    continue;
                }
                ++j;
                while (j < line.size() && (line[j] == ' ' || line[j] == '\t')) {
                    ++j;
                }
                const size_t begin = j;
                while (j < line.size() && (IsAsciiAlnum(line[j]) || line[j] == '-' || line[j] == '_' || line[j] == '.')) {
                    ++j;
                }
                if (j > begin) {
                    return TString(line.substr(begin, j - begin));
                }
            }
            return Nothing();
        };

        TStringBuf rest = source;
        TStringBuf line = rest.NextTok('\n');
        bool comment = false;
        TMaybe<TString> name = spec(line, comment);
        if (!name && comment) {
            name = spec(rest.NextTok('\n'), comment);
        }
        if (!name) {
            return ECookie::None;
        }
        TString norm;
        for (char c : name->substr(0, 12)) {
            norm.push_back(c == '_' ? '-' : AsciiToLower(c));
        }
        if (norm == "utf-8" || norm.StartsWith("utf-8-")) {
            return ECookie::Utf8;
        }
        for (TStringBuf latin : {"latin-1", "iso-8859-1", "iso-latin-1"}) {
            if (norm == latin || norm.StartsWith(TString(latin) + "-")) {
                return ECookie::Latin1;
            }
        }
        return ECookie::Other;
    }

    // The bytecode tools/py3cc used to produce, with its fallback tools/py3cc/slow folded in: a coding
    // cookie naming a codec other than utf-8 or latin-1 over a UTF-8 source is ignored (the source is
    // compiled as UTF-8 text, which is what py3cc/slow did when py3cc failed to find the codec).
    TString Compile(const TString& source, const TString& srcpath) {
        if (source.find('\0') != TString::npos) {
            ythrow yexception() << "failed to compile " << srcpath << ": source code cannot contain null bytes";
        }
        const int cfFlags = CodingCookie(source) == ECookie::Other && IsValidUtf8(source) ? PyCF_IGNORE_COOKIE : 0;

        // arcadia paths are UTF-8 whatever the filesystem encoding of the host is
        TPyObject filename{PyUnicode_DecodeUTF8(srcpath.data(), static_cast<Py_ssize_t>(srcpath.size()), "surrogateescape")};
        if (!filename) {
            PyErr_Print();
            ythrow yexception() << "failed to decode source path " << srcpath;
        }
        PyCompilerFlags flags{cfFlags, PY_MINOR_VERSION};
        TPyObject code{Py_CompileStringObject(source.c_str(), filename.Get(), Py_file_input, &flags, -1)};
        if (!code) {
            PyErr_Print();
            ythrow yexception() << "failed to compile " << srcpath;
        }
        TPyObject bytes{PyMarshal_WriteObjectToString(code.Get(), Py_MARSHAL_VERSION)};
        if (!bytes) {
            PyErr_Print();
            ythrow yexception() << "failed to marshal " << srcpath;
        }
        char* data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(bytes.Get(), &data, &size) != 0) {
            PyErr_Print();
            ythrow yexception() << "failed to marshal " << srcpath;
        }
        return TString(data, static_cast<size_t>(size));
    }

    // --- interpreter ---------------------------------------------------------------------------

    const unsigned char ENCODINGS_ARCHIVE[] = {
#include "pyres_encodings.inc"
    };

    // The encodings package compiled at build time (see ya.make), as PyImport_FrozenModules. With it the
    // interpreter starts normally, with the import system and the codec registry, though there is no
    // standard library around: what the compiler needs besides it is frozen into libpython already
    // (importlib, codecs, io, ...) or built in (unicodedata, for \N{...} and non-ASCII identifiers).
    class TFrozenEncodings {
    public:
        TFrozenEncodings()
            : Archive_(TBlob::NoCopy(ENCODINGS_ARCHIVE, sizeof(ENCODINGS_ARCHIVE)))
        {
            for (size_t i = 0; i < Archive_.Count(); ++i) {
                const TString key = Archive_.KeyByIndex(i);
                TStringBuf name = key;
                name.SkipPrefix("/");
                if (!name.ChopSuffix(".pyc") || !(name == "encodings" || name.StartsWith("encodings."))) {
                    ythrow yexception() << "unexpected entry in the frozen encodings archive: " << key;
                }
                Names_.emplace_back(name);
                Code_.push_back(Archive_.ObjectBlobByKey(key));
            }
            for (size_t i = 0; i < Names_.size(); ++i) {
                Table_.push_back({Names_[i].c_str(), Code_[i].AsUnsignedCharPtr(), static_cast<int>(Code_[i].Size()), Names_[i] == "encodings"});
            }
            Table_.push_back({nullptr, nullptr, 0, 0});
        }

        const _frozen* Table() const noexcept {
            return Table_.data();
        }

    private:
        TArchiveReader Archive_;
        TVector<TString> Names_;
        TVector<TBlob> Code_;
        TVector<_frozen> Table_;
    };

    // The arcadia CPython imports __res (the resource importer of library/python/runtime_py3) when it
    // sets up the import system. There is nothing to import from resources here: an empty module does.
    PyModuleDef EmptyResModule = {PyModuleDef_HEAD_INIT, "__res", nullptr, -1, nullptr, nullptr, nullptr, nullptr, nullptr};

    PyObject* InitEmptyRes() {
        return PyModule_Create(&EmptyResModule);
    }

    // --- pack ----------------------------------------------------------------------------------

    struct TRecord {
        TString Key;
        TString Value;
    };

    template <class T>
    void Put(TString& out, T value) {
        static_assert(std::is_integral_v<T>);
        out.append(reinterpret_cast<const char*>(&value), sizeof(value)); // little-endian hosts only
    }

    void Pad8(TString& out) {
        while (out.size() % 8) {
            out.push_back('\0');
        }
    }

    TString BuildPack(TVector<TRecord>& records) {
        Sort(records, [](const TRecord& a, const TRecord& b) { return a.Key < b.Key; });

        TString out;
        out.append(PACK_MAGIC);
        Put<ui64>(out, 0); // size, patched below
        Put<ui32>(out, static_cast<ui32>(records.size()));
        Put<ui32>(out, 0); // flags
        for (const auto& rec : records) {
            Put<ui32>(out, static_cast<ui32>(rec.Key.size()));
            Put<ui32>(out, static_cast<ui32>(rec.Value.size()));
            out.append(rec.Key);
            out.append(rec.Value);
            Pad8(out);
        }
        const ui64 size = out.size();
        memcpy(out.begin() + PACK_MAGIC.size(), &size, sizeof(size));
        return out;
    }
} // namespace

int main(int argc, char** argv) {
    try {
        const TArgs args = ParseArgs(argv[0], ExpandArgs(argc, argv));

        static const TFrozenEncodings frozenEncodings;
        PyImport_FrozenModules = frozenEncodings.Table();
        if (PyImport_AppendInittab("__res", &InitEmptyRes) != 0) {
            ythrow yexception() << "failed to register __res";
        }

        PyConfig cfg{};
        PyConfig_InitIsolatedConfig(&cfg);
        Y_SCOPE_EXIT(&cfg) {
            PyConfig_Clear(&cfg);
        };
        cfg.site_import = 0;
        cfg.use_frozen_modules = 1;
        cfg.install_signal_handlers = 0;
        cfg.pathconfig_warnings = 0;
        cfg.module_search_paths_set = 1; // empty sys.path: nothing is imported from the file system
        const auto status = Py_InitializeFromConfig(&cfg);
        if (PyStatus_Exception(status)) {
            Py_ExitStatusException(status);
        }
        Y_SCOPE_EXIT() {
            Py_Finalize();
        };

        TVector<TRecord> records;
        for (size_t i = 0; i < args.Src.size(); ++i) {
            const TString& srcx = args.SrcX[i];
            const TString& dest = args.Dest[i];
            const TString& src = args.Src[i];
            if (!srcx.EndsWith('-')) {
                ythrow yexception() << "--srcx value must end with '-': " << srcx;
            }
            const TString rootrel = srcx.substr(0, srcx.size() - 1);
            const TString& srcpath = args.SrcPath[i];
            const TString source = TUnbufferedFileInput(src).ReadAll();

            if (!args.NoPy) {
                records.push_back({"resfs/src/resfs/file/" + dest, NResource::Compress(srcpath)});
                records.push_back({"resfs/file/" + dest, NResource::Compress(source)});
            }
            if (!args.NoPyc) {
                const TString code = Compile(source, rootrel);
                records.push_back({"resfs/src/resfs/file/" + dest + ".yapyc3", NResource::Compress(srcpath + ".yapyc3")});
                records.push_back({"resfs/file/" + dest + ".yapyc3", NResource::Compress(code)});
            }
        }
        for (const TString& kv : args.Kv) {
            const size_t eq = kv.find('=');
            records.push_back({kv.substr(0, eq), NResource::Compress(kv.substr(eq + 1))});
        }

        TFixedBufferFileOutput(args.Output).Write(BuildPack(records));
        return 0;
    } catch (...) {
        Cerr << "pyres_pack: " << CurrentExceptionMessage() << Endl;
        return 1;
    }
}
