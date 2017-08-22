#include <library/json/writer/json_value.h>
#include <library/json/writer/json.h>
#include <library/getopt/small/last_getopt.h>

#include <tools/enum_parser/parse_enum/parse_enum.h>

#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/stream/input.h>
#include <util/stream/mem.h>

#include <util/charset/wide.h>
#include <util/string/strip.h>
#include <util/string/cast.h>
#include <util/string/join.h>
#include <util/string/subst.h>
#include <util/generic/map.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>
#include <util/generic/maybe.h>
#include <util/system/fs.h>
#include <util/folder/path.h>

void WriteHeader(const TString& headerName, IOutputStream& out, IOutputStream* headerOutPtr = nullptr) {
    out << "// This file was auto-generated. Do not edit!!!\n";
    out << "#include " << headerName << "\n";
    out << "#include <util/generic/typetraits.h>\n";
    out << "#include <util/generic/singleton.h>\n";
    out << "#include <util/generic/string.h>\n";
    out << "#include <util/generic/vector.h>\n";
    out << "#include <util/generic/map.h>\n";
    out << "#include <util/generic/serialized_enum.h>\n";
    out << "#include <util/string/cast.h>\n";
    out << "#include <util/stream/output.h>\n\n";
    out << "#include <initializer_list>\n";
    out << "#include <utility>\n\n";

    if (headerOutPtr) {
        auto& outHeader = *headerOutPtr;
        outHeader << "// This file was auto-generated. Do not edit!!!\n";
        outHeader << "#pragma once\n\n";
        outHeader << "#include <util/generic/serialized_enum.h>\n";
        outHeader << "#include " << headerName << "\n";
    }
}

static inline void JsonEscape(TString& s) {
    SubstGlobal(s, "\\", "\\\\");
    SubstGlobal(s, "\"", "\\\"");
    SubstGlobal(s, "\r", "\\r");
    SubstGlobal(s, "\n", "\\n");
    SubstGlobal(s, "\t", "\\t");
}

static inline TString JsonQuote(const TString& s) {
    TString quoted = s;
    JsonEscape(quoted);
    return "\"" + quoted + "\""; // do not use .Quote() here, it performs escaping!
}


/// Simplifed JSON map encoder for generic types
template<typename T>
void OutKey(IOutputStream& out, const TString& key, const T& value, bool escape = true) {
    TString quoted = ToString(value);
    if (escape) {
        quoted = JsonQuote(quoted);
    }
    out << "\"" << key << "\": " << quoted << ",\n";
}

/// Simplifed JSON map encoder for TMaybe
void OutKey(IOutputStream& out, const TString& key, const TMaybe<TString>& value) {
    TString quoted;
    if (value) {
        quoted = JsonQuote(ToString(*value));
    } else {
        quoted = "null";
    }
    out << "\"" << key << "\": " << quoted << ",\n";
}


/// Simplifed JSON map encoder for bool values
void OutKey(IOutputStream& out, const TString& key, const bool& value) {
    out << "\"" << key << "\": " << (value ? "true" : "false") << ",\n";
}


/// Simplifed JSON map encoder for array items
template<typename T>
void OutItem(IOutputStream& out, const T& value, bool escape = true) {
    TString quoted = ToString(value);
    if (escape) {
        quoted = JsonQuote(quoted);
    }
    out << quoted << ",\n";
}

/// Cut trailing ",\n" or ","
static inline void FinishItems(TStringStream& out) {
    TString& s = out.Str();
    if (s.EndsWith(",\n")) {
        s.remove(+s - 2, 2);
    }
    if (s.EndsWith(",")) {
        s.pop_back();
    }
}


static inline void OpenMap(TStringStream& out) {
    out << "{\n";
}

static inline void CloseMap(TStringStream& out) {
    out << "}\n";
}

static inline void OpenArray(TStringStream& out) {
    out << "[\n";
}

static inline void CloseArray(TStringStream& out) {
    out << "]\n";
}

void GenerateEnum(
    const TEnumParser::TEnum& en,
    IOutputStream& out,
    IOutputStream* jsonEnumOut = nullptr,
    IOutputStream* headerOutPtr = nullptr
) {
    TStringStream jEnum;
    OpenMap(jEnum);

    size_t count = +en.Items;
    OutKey(jEnum, "count", count);
    const TString name = TEnumParser::ScopeStr(en.Scope) + en.CppName;
    OutKey(jEnum, "full_name", name);
    OutKey(jEnum, "cpp_name", en.CppName);
    TStringStream scopeJson;
    OpenArray(scopeJson);
    for (const auto& scopeItem : en.Scope) {
        OutItem(scopeJson, scopeItem);
    }
    FinishItems(scopeJson);
    CloseArray(scopeJson);

    OutKey(jEnum, "scope", scopeJson.Str(), false);
    OutKey(jEnum, "enum_class", en.EnumClass);

    TEnumParser::TScope outerScope = en.Scope;
    if (en.EnumClass) {
        outerScope.push_back(en.CppName);
    }

    TString outerScopeStr = TEnumParser::ScopeStr(outerScope);

    TString cName = name;
    SubstGlobal(cName, "::", "");

    out << "// I/O for " << name << "\n";

    TString nsName = "N" + cName + "Private";

    out << "namespace { namespace " << nsName << " {\n";
    out << "    class TNameBufs {\n";
    out << "    private:\n";
    out << "        ymap<" << name << ", TString> Names;\n";
    out << "        ymap<TString, " << name << "> Values;\n";
    out << "        TString AllNames;\n";
    out << "        yvector<" << name << "> AllValues;\n";
    out << "        yvector<TString> AllCppNames;\n";
    out << "    private:\n";
    out << "        inline void AddName(" << name << " key, const TString& strValue) {\n";
    out << "            if (Names.has(key)) {\n";
    out << "                return;\n";
    out << "            }\n";
    out << "            Names[key] = strValue;\n";
    out << "        }\n";
    out << "    public:\n";
    out << "        TNameBufs() {\n";

    yvector<TString> nameInitializerPairs;
    yvector<TString> valueInitializerPairs;
    yvector<TString> cppNamesInitializer;

    TStringStream jItems;
    OpenArray(jItems);

    for (const auto& it : en.Items) {
        TStringStream jEnumItem;
        OpenMap(jEnumItem);

        OutKey(jEnumItem, "cpp_name", it.CppName);
        OutKey(jEnumItem, "value", it.Value);
        OutKey(jEnumItem, "comment_text", it.CommentText);

        TStringStream jAliases;
        OpenArray(jAliases);

        TString strValue = it.CppName;
        if (it.Aliases) {
            // first alias is main
            strValue = it.Aliases[0];
            OutKey(jEnumItem, "str_value", strValue);
        }
        nameInitializerPairs.push_back("{" + outerScopeStr + it.CppName + ", \"" + strValue + "\"}");
        cppNamesInitializer.push_back("\"" + outerScopeStr + it.CppName + "\"");

        for (const auto& alias : it.Aliases) {
            valueInitializerPairs.push_back("{" + outerScopeStr + it.CppName + ", \"" + alias + "\"}");
            OutItem(jAliases, alias);
        }
        FinishItems(jAliases);
        CloseArray(jAliases);

        if (!it.Aliases) {
            valueInitializerPairs.push_back("{" + outerScopeStr + it.CppName + ", \"" + it.CppName + "\"}");
        }
        OutKey(jEnumItem, "aliases", jAliases.Str(), false);

        FinishItems(jEnumItem);
        CloseMap(jEnumItem);

        OutItem(jItems, jEnumItem.Str(), false);
    }
    FinishItems(jItems);
    CloseArray(jItems);
    OutKey(jEnum, "items", jItems.Str(), false);

    out << "            const std::initializer_list<std::pair<" << name << ", const char*>>& namesInitializer = {\n";
    out << "                " << JoinSeq(",\n                ", nameInitializerPairs) << "\n";
    out << "            };\n\n";

    if (count > 0) {
        out << "            const std::initializer_list<const char*>& cppNamesInitializer = {\n";
        out << "                " << JoinSeq(",\n                ", cppNamesInitializer) << "\n";
        out << "            };\n\n";
    }

    if (nameInitializerPairs == valueInitializerPairs) {
        // use the same initializer list if there is no multiple aliases
        out << "            const std::initializer_list<std::pair<" << name << ", const char*>>& valuesInitializer = namesInitializer;\n\n";
    } else {
        out << "            const std::initializer_list<std::pair<" << name << ", const char*>>& valuesInitializer = {\n";
        out << "                " << JoinSeq(",\n                ", valueInitializerPairs) << "\n";
        out << "            };\n\n";
    }

    out << "            for (auto&& it : namesInitializer) {\n";
    out << "                AddName(it.first, it.second);\n";
    out << "            }\n\n";

    out << "            for (auto&& it : valuesInitializer) {\n";
    out << "                Values[it.second] = it.first;\n";
    out << "            }\n\n";

    if (count > 0) {
        out << "            for (auto&& i : Names) {\n";
        out << "                AllNames += \"'\" + i.second + \"', \";\n";
        out << "                AllValues.push_back(i.first);\n";
        out << "            }\n";
        out << "            AllNames = AllNames.substr(0, AllNames.size() - 2);\n";

        // AllCppNames
        out << "            for (auto&& it : cppNamesInitializer) {\n";
        out << "                AllCppNames.push_back(it);\n";
        out << "            }\n\n";
    }

    out << "        }\n\n";
    // ToString
    out << "        const TString& ToString(" << name << " key) const {\n";
    out << "            if (auto pName = Names.FindPtr(key)) {\n";
    out << "                return *pName;\n";
    out << "            }\n";
    // FIXME(mvel): we temporaliy use throw instead of ythrow due to bug DEVTOOLS-3160
    out << "            throw yexception() << \"Undefined value \" << int(key) << \" in " << name << ". \";\n";
    out << "        }\n\n";

    // bool FromString(const TStringBuf& name, <EnumType>& ret)
    out << "        bool FromString(const TStringBuf& name, " << name << "& ret) const {\n";
    out << "            auto it = Values.find(name);\n";
    out << "            if (it != Values.end()) {\n";
    out << "                ret = " << name << "(it->second);\n";
    out << "                return true;\n";
    out << "            }\n";
    out << "            return false;\n";
    out << "        }\n\n";

    // <EnumType> FromString(const TStringBuf& name)
    out << "        " << name << " FromString(const TStringBuf& name) const {\n";
    out << "            " << name << " ret = " << name << "(0);\n";
    out << "            if (FromString(name, ret))\n";
    out << "                return ret;\n";
    out << "            ythrow yexception() << \"Key '\" << name << \"' not found in enum. Valid options are: \" <<\n";
    out << "                AllEnumNames() << \". \";\n";
    out << "        }\n\n";

    // yvector<EnumType> AllEnumValues()
    out << "        const yvector<" << name << ">& AllEnumValues() const {\n";
    out << "            return AllValues;\n";
    out << "        }\n\n";

    // TString AllEnumNames()
    out << "        const TString& AllEnumNames() const {\n";
    out << "            return AllNames;\n";
    out << "        }\n\n";

    // const ymap<EnumType, TString>& EnumNames()
    out << "        const ymap<" << name << ", TString>& EnumNames() const {\n";
    out << "            return Names;\n";
    out << "        }\n\n";

    // TString AllEnumCppNames()
    out << "        const yvector<TString>& AllEnumCppNames() const {\n";
    out << "            return AllCppNames;\n";
    out << "        }\n\n";

    // Instance
    out << "        static inline const TNameBufs& Instance() {\n";
    out << "            return *Singleton<TNameBufs>();\n";
    out << "        }\n";
    out << "    };\n";
    out << "}}\n\n";

    // outer ToString
    out << "const TString& ToString(" << name << " x) {\n";
    out << "    const " << nsName << "::TNameBufs& names = " << nsName << "::TNameBufs::Instance();\n";
    out << "    return names.ToString(x);\n";
    out << "}\n\n";

    // outer FromString
    out << "bool FromString(const TString& name, " << name << "& ret) {\n";
    out << "    const " << nsName << "::TNameBufs& names = " << nsName << "::TNameBufs::Instance();\n";
    out << "    return names.FromString(name, ret);\n";
    out << "}\n\n";

    // outer FromString
    out << "bool FromString(const TStringBuf& name, " << name << "& ret) {\n";
    out << "    const " << nsName << "::TNameBufs& names = " << nsName << "::TNameBufs::Instance();\n";
    out << "    return names.FromString(name, ret);\n";
    out << "}\n\n";

    // specialization for internal FromStringImpl
    out << "template<>\n";
    out << name << " FromStringImpl<" << name << ">(const char* data, size_t len) {\n";
    out << "    const " << nsName << "::TNameBufs& names = " << nsName << "::TNameBufs::Instance();\n";
    out << "    return names.FromString(TStringBuf(data, len));\n";
    out << "}\n\n";

    // specialization for internal TryFromStringImpl
    out << "template<>\n";
    out << "bool TryFromStringImpl<" << name << ">(const char* data, size_t len, " << name << "& result) {\n";
    out << "    return FromString(TStringBuf(data, len), result);\n";
    out << "}\n\n";

    // outer Out
    out << "template<>\n";
    out << "void Out<" << name << ">(IOutputStream& os, TTypeTraits<" << name << ">::TFuncParam n) {\n";
    out << "    os << ToString(n);\n";
    out << "}\n\n";

    // <EnumType>AllValues
    out << "const yvector<" << name << ">& " << cName << "AllValues() {\n";
    out << "    const " << nsName << "::TNameBufs& names = " << nsName << "::TNameBufs::Instance();\n";
    out << "    return names.AllEnumValues();\n";
    out << "}\n\n";

    // <EnumType>AllNames
    out << "const TString& " << cName << "AllNames() {\n";
    out << "    const " << nsName << "::TNameBufs& names = " << nsName << "::TNameBufs::Instance();\n";
    out << "    return names.AllEnumNames();\n";
    out << "}\n\n";

    // template<> GetEnumNames<EnumType>
    out << "template<>\n";
    out << "const ymap<" << name << ", TString>& GetEnumNames<" << name << ">() {\n";
    out << "    const " << nsName << "::TNameBufs& names = " << nsName << "::TNameBufs::Instance();\n";
    out << "    return names.EnumNames();\n";
    out << "}\n\n";

    // <EnumType>AllCppNames, see IGNIETFERRO-534
    out << "const yvector<TString>& " << cName << "AllCppNames() {\n";
    out << "    const " << nsName << "::TNameBufs& names = " << nsName << "::TNameBufs::Instance();\n";
    out << "    return names.AllEnumCppNames();\n";
    out << "}\n\n";

    // <EnumType>FromString
    out << name << " " << cName << "FromString(const TStringBuf& name) {\n";
    out << "    const " << nsName << "::TNameBufs& names = " << nsName << "::TNameBufs::Instance();\n";
    out << "    return names.FromString(name);\n";
    out << "}\n\n";

    if (headerOutPtr) {
        // <EnumType>Count
        auto& outHeader = *headerOutPtr;
        outHeader << "// I/O for " << name << "\n";
        outHeader << "template <>\n";
        outHeader << "constexpr size_t GetEnumItemsCount<" << name << ">() {\n";
        outHeader << "    return " << en.Items.size() << ";\n";
        outHeader << "}\n";
    }

    FinishItems(jEnum);
    jEnum << "}\n";

    if (jsonEnumOut) {
        *jsonEnumOut << jEnum.Str();
    }
}

int main(int argc, char** argv) {
    try {
        using namespace NLastGetopt;
        TOpts opts = NLastGetopt::TOpts::Default();
        opts.AddHelpOption();

        TString outputFileName;
        TString outputHeaderFileName;
        TString outputJsonFileName;
        TString includePath;
        opts.AddLongOption('o', "output").OptionalArgument("<output-file>").StoreResult(&outputFileName)
            .Help(
                "Output generated code to specified file.\n"
                "When not set, standard output is used."
            );
        opts.AddLongOption('h', "header").OptionalArgument("<output-header>").StoreResult(&outputHeaderFileName)
            .Help(
                "Generate appropriate header to specified file.\n"
                "Works only if output file specified."
            );
        opts.AddLongOption("include-path").OptionalArgument("<header-path>").StoreResult(&includePath)
            .Help(
                "Include input header using this path in angle brackets.\n"
                "When not set, header basename is used in double quotes."
            );

        opts.AddLongOption('j', "json-output").OptionalArgument("<json-output>").StoreResult(&outputJsonFileName)
            .Help(
                "Generate enum data in JSON format."
            );

        opts.SetFreeArgsNum(1);
        opts.SetFreeArgTitle(0, "<input-file>", "Input header file with enum declarations");

        TOptsParseResult res(&opts, argc, argv);

        yvector<TString> freeArgs = res.GetFreeArgs();
        TString inputFileName = freeArgs[0];

        THolder<IOutputStream> hOut;
        IOutputStream* out = &Cout;

        THolder<IOutputStream> headerOut;

        THolder<IOutputStream> jsonOut;


        if (outputFileName) {
            NFs::Remove(outputFileName);
            hOut.Reset(new TAdaptiveFileOutput(outputFileName));
            out = hOut.Get();

            if (outputHeaderFileName) {
                headerOut.Reset(new TAdaptiveFileOutput(outputHeaderFileName));
            }

            if (outputJsonFileName) {
                jsonOut.Reset(new TAdaptiveFileOutput(outputJsonFileName));
            }
        }

        if (!includePath) {
            includePath = TString() + '"' + TFsPath(inputFileName).Basename() + '"';
        } else {
            includePath = TString() + '<' + includePath + '>';
        }

        TEnumParser parser(inputFileName);
        WriteHeader(includePath, *out, headerOut.Get());

        TStringStream jEnums;
        OpenArray(jEnums);

        for (const auto& en : parser.Enums) {
            if (!en.CppName) {
                // skip unnamed enum declarations
                continue;
            }

            TStringStream jEnum;
            GenerateEnum(en, *out, &jEnum, headerOut.Get());
            OutItem(jEnums, jEnum.Str(), false);
        }
        FinishItems(jEnums);
        CloseArray(jEnums);

        if (jsonOut) {
            *jsonOut << jEnums.Str() << Endl;
        }

        return 0;
    } catch (...) {
        Cerr << CurrentExceptionMessage() << Endl;
    }

    return 1;
}
