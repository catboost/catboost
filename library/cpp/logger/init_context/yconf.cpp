#include "yconf.h"

TLogBackendCreatorInitContextYConf::TLogBackendCreatorInitContextYConf(const TYandexConfig::Section& section)
    : Section(section)
{}

bool TLogBackendCreatorInitContextYConf::GetValue(TStringBuf name, TString& var) const {
    return Section.GetDirectives().GetValue(name, var);
}

TVector<THolder<ILogBackendCreator::IInitContext>> TLogBackendCreatorInitContextYConf::GetChildren(TStringBuf name) const {
    TVector<THolder<IInitContext>> result;
    auto children = Section.GetAllChildren();
    for (auto range = children.equal_range(TCiString(name)); range.first != range.second; ++range.first) {
        result.emplace_back(MakeHolder<TLogBackendCreatorInitContextYConf>(*range.first->second));
    }
    return result;
}
