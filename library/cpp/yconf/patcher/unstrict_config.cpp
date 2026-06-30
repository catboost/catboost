#include "unstrict_config.h"

#include <util/string/util.h>

namespace {
    template <class T, class TStringType>
    T FromStringWithDefaultForEmpty(const TStringType s, const T& def) {
        if (s) {
            return FromString<T>(s);
        } else {
            return def;
        }
    }
}

const TYandexConfig::Section* TUnstrictConfig::GetSection(const TString& path) {
    TVector<TString> vPath = SplitString(path, ".");
    return GetSection(GetRootSection(), vPath.begin(), vPath.end());
}

bool TUnstrictConfig::AddSection(const TString& path) {
    TVector<TString> vPath = SplitString(path, ".");
    if (GetSection(GetRootSection(), vPath.begin(), vPath.end()))
        return false;
    GetSection(GetRootSection(), vPath.begin(), vPath.end(), true);
    return true;
}

TString TUnstrictConfig::GetValue(const TString& path) const {
    try {
        TVector<TString> vPath = SplitString(path, ".");
        TVector<TString>::const_iterator dir = vPath.end() - 1;
        const TYandexConfig::Section* section = GetSection(GetRootSection(), vPath.begin(), dir);
        if (!section)
            return "__not_found__";
        const Directives& directives = section->GetDirectives();
        TString result;
        if (!directives.GetValue(*dir, result))
            return "__not_found__";
        return result;
    } catch (const yexception& e) {
        ythrow yexception() << "Error while parse path " << path << ": " << e.what();
    }
}

bool TUnstrictConfig::SetValue(const TString& path, const TString& value) {
    try {
        TVector<TString> vPath = SplitString(path, ".");
        TVector<TString>::const_iterator dir = vPath.end() - 1;
        auto sections = GetSections(GetRootSection(), vPath.begin(), dir, true);
        bool result = false;
        for (auto&& section : sections) {
            Directives& directives = section->GetDirectives();
            Directives::iterator iter = directives.find(*dir);
            if (iter == directives.end()) {
                Strings.push_back(value);
                directives.insert(Directives::value_type(*dir, Strings.back().data()));
                result = true;
                continue;
            };
            if (!TCiString::compare(iter->second, value))
                continue;
            Strings.push_back(value);
            iter->second = Strings.back().data();
            result = true;
        }
        return result;
    } catch (const yexception& e) {
        ythrow yexception() << "Error while parse path " << path << ": " << e.what();
    }
}

bool TUnstrictConfig::Remove(const TString& path) {
    TVector<TString> vPath = SplitString(path, ".");
    TVector<TString>::const_iterator dir = vPath.end() - 1;
    TVector<TYandexConfig::Section*> sections = GetSections(GetRootSection(), vPath.begin(), dir, false);
    bool result = false;
    for (auto&& section : sections) {
        result = RemoveSection(*section, *dir) || RemoveDirective(*section, *dir);
    }
    return result;
}

bool TUnstrictConfig::RemoveAll(const TString& path) {
    TVector<TString> vPath = SplitString(path, ".");
    TVector<TString>::const_iterator dir = vPath.end() - 1;
    TVector<TYandexConfig::Section*> sections = GetSections(GetRootSection(), vPath.begin(), dir, false);
    bool result = false;
    for (auto&& section : sections) {
        result = RemoveAllSections(*section, *dir);
    }
    return result;
}

void TUnstrictConfig::ToJsonPatch(const Section& section, NJson::TJsonValue& result, const TString& preffix) {
    const TString& newPreffix = preffix ? preffix + "." : TString();
    for (const auto& i : section.GetDirectives())
        result.InsertValue(newPreffix + i.first, i.second);
    TSectionsMap children = section.GetAllChildren();
    for (TSectionsMap::const_iterator i = children.begin(), e = children.end(); i != e; ++i)
        ToJsonPatch(*i->second, result, newPreffix + i->first);
}

bool TUnstrictConfig::OnBeginSection(Section& sec) {
    sec.Cookie = new AnyDirectives;
    sec.Owner = true;
    if (!sec.Parent->Parsed())
        return true;
    ReportError(sec.Name, true, "section \'%s\' not allowed here and will be ignored", sec.Name);
    return true;
}

TUnstrictConfig::TPathUnit TUnstrictConfig::ProcessPathUnit(const TString& element) const {
    size_t indexBeginPos = element.find('[');
    if (indexBeginPos == TString::npos) {
        return TPathUnit(element, 0);
    }

    size_t indexEndPos = element.find(']', indexBeginPos);
    if (indexEndPos == TString::npos) {
        ythrow yexception() << "Syntax error: ] symbol expected: " << element;
    }

    const TCiString name(element.data(), indexBeginPos);
    const TStringBuf range(element.data() + indexBeginPos + 1, indexEndPos - indexBeginPos - 1);
    if (range.find(':') == range.npos) {
        const size_t index = FromString<size_t>(range);
        return TPathUnit(name, index);
    } else {
        const size_t startIndex = FromStringWithDefaultForEmpty<size_t>(range.Before(':'), 0);
        const size_t endIndex = FromStringWithDefaultForEmpty<size_t>(range.After(':'), Max<size_t>());
        if (startIndex > endIndex) {
            ythrow yexception() << "Incorrect range " << range;
        }
        return TPathUnit(name, startIndex, endIndex);
    }
}

const TYandexConfig::Section* TUnstrictConfig::GetSection(const TYandexConfig::Section* section, const TVector<TString>::const_iterator& begin, const TVector<TString>::const_iterator& end) const {
    auto sections = GetSections(section, begin, end);
    if (sections.empty()) {
        return nullptr;
    } else if (sections.size() == 1) {
        return sections.front();
    } else {
        ythrow yexception() << "more than one sections matched: " << sections.size();
    }
}

TYandexConfig::Section* TUnstrictConfig::GetSection(TYandexConfig::Section* section, const TVector<TString>::const_iterator& begin, const TVector<TString>::const_iterator& end, bool add) {
    auto sections = GetSections(section, begin, end, add);
    if (sections.empty()) {
        return nullptr;
    } else if (sections.size() == 1) {
        return sections.front();
    } else {
        ythrow yexception() << "more than one sections matched: " << sections.size();
    }
}

TVector<const TYandexConfig::Section*> TUnstrictConfig::GetSections(const TYandexConfig::Section* section, TPathIterator begin, TPathIterator end) const {
    Y_ABORT_UNLESS(section);
    if (begin == end)
        return {section};

    TPathUnit pu = ProcessPathUnit(*begin);
    TSectionsMap sections = section->GetAllChildren();
    std::pair<TSectionsMap::const_iterator, TSectionsMap::const_iterator> range = sections.equal_range(pu.Name);

    TVector<const TYandexConfig::Section*> intermediate;
    size_t index = 0;
    for (auto i = range.first; i != range.second && index <= pu.EndIndex; ++i, ++index) {
        intermediate.push_back(i->second);
    }

    TVector<const TYandexConfig::Section*> result;
    for (auto i = pu.BeginIndex; i <= pu.EndIndex && i < intermediate.size(); ++i) {
        auto is = GetSections(intermediate[i], begin + 1, end);
        result.insert(result.end(), is.begin(), is.end());
    }

    return result;
}

TVector<TYandexConfig::Section*> TUnstrictConfig::GetSections(TYandexConfig::Section* section, TPathIterator begin, TPathIterator end, bool add) {
    Y_ABORT_UNLESS(section);
    if (begin == end)
        return {section};

    TPathUnit pu = ProcessPathUnit(*begin);
    TSectionsMap sections = section->GetAllChildren();
    std::pair<TSectionsMap::const_iterator, TSectionsMap::const_iterator> range = sections.equal_range(pu.Name);

    TVector<TYandexConfig::Section*> intermediate;
    size_t index = 0;
    for (auto i = range.first; i != range.second && index <= pu.EndIndex; ++i, ++index) {
        intermediate.push_back(i->second);
    }

    if (add && pu.EndIndex != Max<size_t>()) {
        for (; index <= pu.EndIndex; ++index) {
            auto next = new Section();
            Strings.push_back(pu.Name);
            next->Name = Strings.back().data();
            next->Cookie = new AnyDirectives;
            next->Owner = true;
            next->Parent = section;
            TYandexConfig::AddSection(next);
            intermediate.push_back(next);
        }
    }

    TVector<TYandexConfig::Section*> result;
    for (auto i = pu.BeginIndex; i <= pu.EndIndex && i < intermediate.size(); ++i) {
        auto is = GetSections(intermediate[i], begin + 1, end, add);
        result.insert(result.end(), is.begin(), is.end());
    }
    return result;
}

bool TUnstrictConfig::RemoveSection(Section& parent, const TString& name) {
    TPathUnit pu = ProcessPathUnit(name);
    TYandexConfig::Section* prevSection = nullptr;
    TYandexConfig::Section* curSection = parent.Child;
    size_t index = 0;
    bool deleted = false;
    while (curSection) {
        if (pu.Name == curSection->Name) {
            if (index >= pu.BeginIndex && index <= pu.EndIndex) {
                if (prevSection)
                    prevSection->Next = curSection->Next;
                if (parent.Child == curSection)
                    parent.Child = curSection->Next;
                curSection = curSection->Next;
                deleted = true;
            } else {
                prevSection = curSection;
                curSection = curSection->Next;
            }

            ++index;
            if (index > pu.EndIndex) {
                break;
            }
        } else {
            prevSection = curSection;
            curSection = curSection->Next;
        }
    }
    return deleted;
}

bool TUnstrictConfig::RemoveAllSections(Section& parent, const TString& name) {
    TPathUnit pu = ProcessPathUnit(name);
    if (pu.BeginIndex) {
        ythrow yexception() << "incorrect path for RemoveAllSections: " << name;
    }

    ui32 deleted = 0;
    while (RemoveSection(parent, name)) {
        deleted++;
    }
    return deleted;
}

bool TUnstrictConfig::RemoveDirective(Section& parent, const TString& name) {
    Directives::iterator i = parent.GetDirectives().find(name);
    if (i == parent.GetDirectives().end())
        return false;
    parent.Cookie->erase(i);
    return true;
}

bool TUnstrictConfig::PatchEntry(const TString& path, const TString& value, const TString& prefix /* = "" */) {
    if (value == "__remove__")
        return Remove(prefix + path);
    else if (value == "__remove_all__")
        return RemoveAll(prefix + path);
    else if (value == "__add_section__")
        return AddSection(prefix + path);
    else
        return SetValue(prefix + path, value);
}

[[nodiscard]] bool TUnstrictConfig::ParseJson(const NJson::TJsonValue& json) {
    Y_ABORT_UNLESS(ParseMemory(""));
    return ParseJson(json, TString());
}

[[nodiscard]] bool TUnstrictConfig::ParseJson(const NJson::TJsonValue& json, const TString& path) {
    if (json.IsNull()) {
        return true;
    }
    if (!json.IsMap()) {
        const TString& error = (path ? path : "Root") + "section must be a Json map";
        ReportError(nullptr, error.data());
        return false;
    }
    for (auto&& e : json.GetMap()) {
        const TString& name = e.first;
        const NJson::TJsonValue& node = e.second;

        if (node.IsArray()) {
            size_t index = 0;
            for (auto&& element : node.GetArray()) {
                AddSection(path + name);
                if (!ParseJson(element, path + name + "[" + ::ToString(index) + "].")) {
                    return false;
                }
                ++index;
            }
        } else if (node.IsMap()) {
            const TString& error = path + name + " section must be either a Json array or a Json plain value";
            ReportError(nullptr, error.data());
            return false;
        } else {
            SetValue(path + name, node.GetStringRobust());
        }
    }
    return true;
}

TString TUnstrictConfig::ToString() const {
    TStringStream stringStream;
    const Section* root = GetRootSection();
    if (!root->Child)
        ythrow yexception() << "root element of the config has no children";
    SectionToStream(root->Child, stringStream, 0);
    return stringStream.Str();
}

NJson::TJsonValue TUnstrictConfig::ToJson() const {
    return ToJson(*this);
}

void TUnstrictConfig::ToJson(const Section& section, NJson::TJsonValue& result) {
    for (const auto& i : section.GetDirectives()) {
        result[i.first] = i.second;
    }
    TSectionsMap children = section.GetAllChildren();
    for (TSectionsMap::const_iterator i = children.begin(), e = children.end(); i != e; ++i)
        result[i->first].AppendValue(ToJson(*i->second));
}

void TUnstrictConfig::ToJson(const TYandexConfig& config, NJson::TJsonValue& result) {
    return ToJson(*config.GetRootSection(), result);
}

void TUnstrictConfig::ToJson(const TString& config, NJson::TJsonValue& result) {
    TUnstrictConfig uc;
    if (!uc.ParseMemory(config.data())) {
        TString errors;
        uc.PrintErrors(errors);
        throw yexception() << "Cannot parse YandexConfig: " << errors;
    }
    ToJson(uc, result);
}

void SectionToStream(const TYandexConfig::Section* section, IOutputStream& stream, ui16 level) {
    TString shift;
    shift.reserve(level * 4);
    for (int i = 0; i < level; ++i)
        shift += "    ";
    TString childShift = shift + "    ";
    if (!section || !section->Parsed())
        return;
    bool hasName = section->Name && *section->Name;
    if (hasName) {
        stream << shift << "<" << section->Name;
        for (const auto& attr : section->Attrs) {
            stream << " " << attr.first << "=\"" << attr.second << "\"";
        }
        stream << ">\n";
    }
    for (const auto& iter : section->GetDirectives()) {
        stream << childShift << iter.first;
        stream << " : " << TString(iter.second) << "\n";
    }
    if (section->Child) {
        SectionToStream(section->Child, stream, level + 1);
    }
    if (hasName)
        stream << shift << "</" << section->Name << ">\n";

    if (section->Next) {
        SectionToStream(section->Next, stream, level);
    }
}
