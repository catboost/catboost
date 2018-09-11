#include "detail.h"

namespace NCB {
    namespace NQuantizationSchemaDetail {
        bool IsFakeIndex(ui32 index, const NIdl::TPoolMetainfo& metaInfo) {
            return (
                (metaInfo.HasStringDocIdFakeColumnIndex() && index == metaInfo.GetStringDocIdFakeColumnIndex()) ||
                (metaInfo.HasStringDocIdFakeColumnIndex() && index == metaInfo.GetStringGroupIdFakeColumnIndex()) ||
                (metaInfo.HasStringDocIdFakeColumnIndex() && index == metaInfo.GetStringSubgroupIdFakeColumnIndex())
            );
        }
    }
}
