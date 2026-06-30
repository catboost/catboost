#include "config_patcher.h"

#include <library/cpp/yconf/patcher/unstrict_config.h>

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_prettifier.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/xrange.h>
#include <util/stream/file.h>
#include <util/string/subst.h>

#include <array>

namespace {
    class TWrapper {
    public:
        TWrapper(const TString& base, const NJson::TJsonValue& patch, const TString& prefix);

        void Patch();
        TString GetPatchedConfig() {
            Patch();
            return PatchedConfixText;
        }

    private:
        void Preprocess();
        void Postprocess();

    private:
        static const TString IncludeDirective;
        static const TString IncludeGuard;

    private:
        TString BaseConfigText;
        THolder<TUnstrictConfig> Config;
        NJson::TJsonValue JsonPatch;
        TString Prefix;
        TString PatchedConfixText;
        bool Patched;
        bool NeedsPostprocessing;
    };

    const TString TWrapper::IncludeDirective = "#include";
    const TString TWrapper::IncludeGuard = "__INCLUDE__ :";

    TWrapper::TWrapper(const TString& base, const NJson::TJsonValue& patch, const TString& prefix)
        : BaseConfigText(base)
        , JsonPatch(patch)
        , Prefix(prefix)
        , Patched(false)
        , NeedsPostprocessing(false)
    {
    }

    void TWrapper::Patch() {
        if (Patched)
            return;
        Preprocess();

        const NJson::TJsonValue::TMapType* values;
        if (!JsonPatch.GetMapPointer(&values))
            ythrow yexception() << "unable to get map pointer in the json patch.";
        for (const auto& value : *values) {
            Config->PatchEntry(value.first, value.second.GetStringRobust(), Prefix);
        }

        Postprocess();
        Patched = true;
    }

    void TWrapper::Preprocess() {
        TString parsed;
        size_t pos = BaseConfigText.find(IncludeDirective);
        if (pos != TString::npos) {
            NeedsPostprocessing = true;
            parsed = BaseConfigText.replace(pos, IncludeDirective.size(), IncludeGuard);
        } else {
            parsed = BaseConfigText;
        }
        Config.Reset(new TUnstrictConfig);
        if (!Config->ParseMemory(parsed.data())) {
            TString errors;
            Config->PrintErrors(errors);
            ythrow yexception() << "Can't parse config:" << errors;
        }
    }

    void TWrapper::Postprocess() {
        PatchedConfixText = Config->ToString();
        if (NeedsPostprocessing) {
            SubstGlobal(PatchedConfixText, IncludeGuard, IncludeDirective);
        }
    }

    void MakeDiff(
            NJson::TJsonValue& container,
            const TYandexConfig::Section* source,
            const TYandexConfig::Section* target,
            const TString& parentPrefix = TString()) {
        Y_ABORT_UNLESS(target);
        const TString& prefix = parentPrefix ? (parentPrefix + ".") : parentPrefix;

        for (const auto& [name, value]: target->GetDirectives()) {
            const auto p = source ? source->GetDirectives().FindPtr(name) : nullptr;
            if (!p || TString(*p) != value) {
                container[prefix + name] = value;
            }
        }

        if (source) {
            for (const auto& [name, value] : source->GetDirectives()) {
                if (!target->GetDirectives().contains(name)) {
                    container[prefix + name] = "__remove__";
                }
            }
        }

        TMap<TCiString, std::array<TVector<const TYandexConfig::Section *>, 2>> alignedSections;

        auto fillSections = [&](const TYandexConfig::TSectionsMap& sections, size_t targetIndex) {
            for (const auto& [sectionName, section] : sections) {
                alignedSections[sectionName][targetIndex].push_back(section);
            }
        };

        if (source) {
            fillSections(source->GetAllChildren(), 0);
        }
        fillSections(target->GetAllChildren(), 1);

        for (const auto& [sectionName, pair]: alignedSections) {
            // Cannot use structured binding on std::array here because of a bug in MSVC.
            const auto& sourceSections = pair[0];
            const auto& targetSections = pair[1];
            const bool needsIndex = targetSections.size() > 1;

            if (targetSections.empty()) {
                Y_ABORT_UNLESS(source);
                container[prefix + sectionName] = "__remove_all__";
            } else {
                for (const size_t i: xrange(targetSections.size())) {
                    MakeDiff(
                            container,
                            i < sourceSections.size() ? sourceSections[i] : nullptr,
                            targetSections[i],
                            !needsIndex ? prefix + sectionName : prefix + sectionName + '[' + ToString(i) + ']');
                }
                if (sourceSections.size() > targetSections.size()) {
                    container[
                        prefix +
                            sectionName +
                            '[' +
                            ToString(targetSections.size()) +
                            ':' +
                            ToString(sourceSections.size() - 1) +
                            ']'] = "__remove__";
                }
            }
        }

        if (target->GetAllChildren().empty() && target->GetDirectives().empty()) {
            container[prefix] = "__add_section__";
        }
    }
}

namespace NConfigPatcher {
    TString Patch(const TString& config, const TString& patch, const TOptions& options) {
        if (!patch) {
            return config;
        }

        NJson::TJsonValue parsedPatch;
        TStringInput ss(patch);
        if (!NJson::ReadJsonTree(&ss, true, &parsedPatch, true)) {
            ythrow yexception() << "Cannot parse patch as json";
        }
        return Patch(config, parsedPatch, options);
    }

    TString Patch(const TString& config, const NJson::TJsonValue& parsedPatch, const TOptions& options) {
        TWrapper patcher(config, parsedPatch, options.Prefix);
        return patcher.GetPatchedConfig();
    }

    TString Diff(const TString& sourceText, const TString& targetText) {
        TUnstrictConfig source;
        if (!source.ParseMemory(sourceText.data())) {
            throw yexception() << "Cannot parse source config";
        }

        TUnstrictConfig target;
        if (!target.ParseMemory(targetText.data())) {
            throw yexception() << "Cannot parse target config";
        }

        NJson::TJsonValue diff(NJson::JSON_MAP);
        MakeDiff(diff, source.GetRootSection(), target.GetRootSection());
        return NJson::PrettifyJson(diff.GetStringRobust());
    }
}
