#include <library/cpp/testing/benchmark/bench.h>

#include <library/cpp/containers/comptrie/comptrie_trie.h>
#include <library/cpp/containers/comptrie/comptrie_builder.h>
#include <library/cpp/containers/comptrie/search_iterator.h>
#include <library/cpp/containers/comptrie/pattern_searcher.h>

#include <library/cpp/on_disk/aho_corasick/writer.h>
#include <library/cpp/on_disk/aho_corasick/reader.h>
#include <library/cpp/on_disk/aho_corasick/helpers.h>

#include <library/cpp/containers/dense_hash/dense_hash.h>

#include <util/stream/file.h>
#include <util/generic/algorithm.h>
#include <util/random/fast.h>
#include <util/random/shuffle.h>

/////////////////
// COMMON DATA //
/////////////////

const size_t MAX_PATTERN_LENGTH = 11;

TVector<TString> letters = {
    "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й",
    "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф",
    "х", "ц", "ч", "ж", "щ", "ъ", "ы", "ь", "э", "ю", "я"
};

TString GenerateOneString(
    TFastRng<ui64>& rng,
    size_t maxLength,
    const TVector<TString>& sequences
) {
    size_t length = rng.GenRand() % maxLength + 1;
    TString result;
    while (result.size() < length) {
        result += sequences[rng.GenRand() % sequences.size()];
    }
    return result;
}

TVector<TString> GenerateStrings(
    TFastRng<ui64>& rng,
    size_t num,
    size_t maxLength,
    const TVector<TString>& sequences
) {
    TVector<TString> strings;
    while (strings.size() < num) {
        strings.push_back(GenerateOneString(rng, maxLength, sequences));
    }
    return strings;
}

struct TDatasetInstance {
    TDatasetInstance(const TVector<TString>& sequences) {
        TFastRng<ui64> rng(0);

        TVector<TString> prefixes = GenerateStrings(rng, /*num*/10, /*maxLength*/3, sequences);
        prefixes.push_back("");

        TVector<TString> roots = GenerateStrings(rng, /*num*/1000, /*maxLength*/5, sequences);

        TVector<TString> suffixes = GenerateStrings(rng, /*num*/10, /*maxLength*/3, sequences);
        suffixes.push_back("");

        TVector<TString> dictionary;
        for (const auto& root : roots) {
            for (const auto& prefix : prefixes) {
                for (const auto& suffix : suffixes) {
                    dictionary.push_back(prefix + root + suffix);
                    Y_ASSERT(dictionary.back().size() < MAX_PATTERN_LENGTH);
                }
            }
        }
        Shuffle(dictionary.begin(), dictionary.end());

        Patterns.assign(dictionary.begin(), dictionary.begin() + 10'000);

        for (size_t sampleIdx = 0; sampleIdx < /*samplesNum*/1'000'000; ++sampleIdx) {
            Samples.emplace_back();
            size_t wordsNum = rng.GenRand() % 10;
            for (size_t wordIdx = 0; wordIdx < wordsNum; ++wordIdx) {
                if (wordIdx > 0) {
                    Samples.back() += " ";
                }
                Samples.back() += dictionary[rng.GenRand() % dictionary.size()];
            }
        }
    }

    TString GetSample(size_t iteration) const {
        TFastRng<ui64> rng(iteration);
        return Samples[rng.GenRand() % Samples.size()];
    }


    TVector<TString> Patterns;
    TVector<TString> Samples;
};

static const TDatasetInstance dataset(letters);

//////////////////////////
// NEW PATTERN SEARCHER //
//////////////////////////

struct TPatternSearcherInstance {
    TPatternSearcherInstance() {
        TCompactPatternSearcherBuilder<char, ui32> builder;

        for (ui32 patternId = 0; patternId < dataset.Patterns.size(); ++patternId) {
            builder.Add(dataset.Patterns[patternId], patternId);
        }

        TBufferOutput buffer;
        builder.Save(buffer);

        Instance.Reset(
            new TCompactPatternSearcher<char, ui32>(
                buffer.Buffer().Data(),
                buffer.Buffer().Size()
            )
        );
    }

    THolder<TCompactPatternSearcher<char, ui32>> Instance;
};

static const TPatternSearcherInstance patternSearcherInstance;

Y_CPU_BENCHMARK(PatternSearcher, iface) {
    TVector<TVector<std::pair<ui32, ui32>>> result;
    for (size_t iteration = 0; iteration < iface.Iterations(); ++iteration) {
        result.emplace_back();
        TString testString = dataset.GetSample(iteration);
        auto matches = patternSearcherInstance.Instance->SearchMatches(testString);
        for (auto& match : matches) {
            result.back().emplace_back(match.End, match.Data);
        }
    }
}

//////////////////////
// OLD AHO CORASICK //
//////////////////////

struct TAhoCorasickInstance {
    TAhoCorasickInstance() {
        TAhoCorasickBuilder<TString, ui32> builder;

        for (ui32 patternId = 0; patternId < dataset.Patterns.size(); ++patternId) {
            builder.AddString(dataset.Patterns[patternId], patternId);
        }

        TBufferOutput buffer;
        builder.SaveToStream(&buffer);

        Instance.Reset(new TDefaultMappedAhoCorasick(TBlob::FromBuffer(buffer.Buffer())));
    }

    THolder<TDefaultMappedAhoCorasick> Instance;
};

static const TAhoCorasickInstance ahoCorasickInstance;

Y_CPU_BENCHMARK(AhoCorasick, iface) {
    TVector<TDeque<std::pair<ui32, ui32>>> result;
    for (size_t iteration = 0; iteration < iface.Iterations(); ++iteration) {
        result.emplace_back();
        TString testString = dataset.GetSample(iteration);
        auto matches = ahoCorasickInstance.Instance->AhoSearch(testString);
        result.push_back(matches);
    }
}

////////////////////////////////
// COMPTRIE + SIMPLE MATCHING //
////////////////////////////////

struct TCompactTrieInstance {
    TCompactTrieInstance() {
        TCompactTrieBuilder<char, ui32> builder;

        for (ui32 patternId = 0; patternId < dataset.Patterns.size(); ++patternId) {
            builder.Add(dataset.Patterns[patternId], patternId);
        }


        TBufferOutput buffer;
        CompactTrieMinimizeAndMakeFastLayout(buffer, builder);

        Instance.Reset(new TCompactTrie<char, ui32>(
            buffer.Buffer().Data(),
            buffer.Buffer().Size()
        ));
    }

    THolder<TCompactTrie<char, ui32>> Instance;
};

static const TCompactTrieInstance compactTrieInstance;

Y_CPU_BENCHMARK(ComptrieSimple, iface) {
    TVector<TVector<std::pair<ui32, ui32>>> result;
    for (size_t iteration = 0; iteration < iface.Iterations(); ++iteration) {
        result.emplace_back();
        TString testString = dataset.GetSample(iteration);
        for (ui32 startPos = 0; startPos < testString.size(); ++startPos) {
            TSearchIterator<TCompactTrie<char, ui32>> iter(*(compactTrieInstance.Instance));
            for (ui32 position = startPos; position < testString.size(); ++position) {
                if (!iter.Advance(testString[position])) {
                    break;
                }
                ui32 answer;
                if (iter.GetValue(&answer)) {
                    result.back().emplace_back(position, answer);
                }
            }
        }
    }
}

////////////////
// DENSE_HASH //
////////////////

struct TDenseHashInstance {
    TDenseHashInstance() {
        for (ui32 patternId = 0; patternId < dataset.Patterns.size(); ++patternId) {
            Instance[dataset.Patterns[patternId]] = patternId;
        }
    }

    TDenseHash<TString, ui32> Instance;
};

static const TDenseHashInstance denseHashInstance;

Y_CPU_BENCHMARK(DenseHash, iface) {
    TVector<TVector<std::pair<ui32, ui32>>> result;
    for (size_t iteration = 0; iteration < iface.Iterations(); ++iteration) {
        result.emplace_back();
        TString testString = dataset.GetSample(iteration);
        for (size_t start = 0; start < testString.size(); ++start) {
            for (
                size_t length = 1;
                length <= MAX_PATTERN_LENGTH && start + length <= testString.size();
                ++length
            ) {
                auto value = denseHashInstance.Instance.find(testString.substr(start, length));
                if (value != denseHashInstance.Instance.end()) {
                    result.back().emplace_back(start + length - 1, value->second);
                }
            }
        }
    }
}
