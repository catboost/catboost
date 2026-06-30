#include <library/cpp/containers/flat_hash/lib/map.h>
#include <library/cpp/containers/flat_hash/lib/containers.h>
#include <library/cpp/containers/flat_hash/lib/probings.h>
#include <library/cpp/containers/flat_hash/lib/size_fitters.h>
#include <library/cpp/containers/flat_hash/lib/expanders.h>

#include <library/cpp/containers/flat_hash/fuzz/fuzz_common/fuzz_common.h>

#include <util/generic/hash.h>
#include <util/generic/xrange.h>
#include <util/generic/yexception.h>

using namespace NFlatHash;

namespace {

template <class Key, class T>
using TDenseModMap = NFlatHash::TMap<Key,
                                     T,
                                     THash<Key>,
                                     std::equal_to<Key>,
                                     TRemovalDenseContainer<std::pair<const Key, T>,
                                                            NMap::TEqValueMarker<Key, T>,
                                                            NMap::TEqValueMarker<Key, T>>,
                                     TDenseProbing,
                                     TAndSizeFitter,
                                     TSimpleExpander>;

NFuzz::EActionType EvalType(ui8 data) {
    return static_cast<NFuzz::EActionType>((data >> 5) & 0b111);
}

ui8 EvalKey(ui8 data) {
    return data & 0b11111;
}

ui8 EvalValue() {
    return RandomNumber<ui8>();
}

}  // namespace

#include <util/datetime/base.h>

extern "C" int LLVMFuzzerTestOneInput(const ui8* const wireData, const size_t wireSize) {
    THashMap<ui8, ui8> etalon;
    // We assume, that markers can't be produced by EvalKey function.
    TDenseModMap<ui8, ui8> testee(8,
                                  (1 << 5), // Deleted marker
                                  (1 << 6)); // Empty marker

    for (auto i : xrange(wireSize)) {
        auto data = wireData[i];

        NFuzz::MakeAction(etalon, testee, EvalKey(data), EvalValue(), EvalType(data));
        NFuzz::CheckInvariants(etalon, testee);
    }

    return 0;
}
