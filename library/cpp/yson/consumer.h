#pragma once

#include <library/cpp/yt/yson/consumer.h>

#include <util/generic/strbuf.h>
#include <util/system/defaults.h>

namespace NYson {
    struct TYsonConsumerBase
       : public virtual NYT::NYson::IYsonConsumer {
        void OnRaw(TStringBuf ysonNode, NYT::NYson::EYsonType type) override;
    };
} // namespace NYson
