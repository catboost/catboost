#pragma once

#include "public.h"

#include <util/generic/strbuf.h>
#include <util/system/defaults.h>

namespace NYT {
    ////////////////////////////////////////////////////////////////////////////////

    struct IYsonConsumer {
        virtual ~IYsonConsumer() {
        }

        virtual void OnStringScalar(const TStringBuf& value) = 0;
        virtual void OnInt64Scalar(i64 value) = 0;
        virtual void OnUint64Scalar(ui64 scalar) = 0;
        virtual void OnDoubleScalar(double value) = 0;
        virtual void OnBooleanScalar(bool value) = 0;

        virtual void OnEntity() = 0;

        virtual void OnBeginList() = 0;
        virtual void OnListItem() = 0;
        virtual void OnEndList() = 0;

        virtual void OnBeginMap() = 0;
        virtual void OnKeyedItem(const TStringBuf& key) = 0;
        virtual void OnEndMap() = 0;

        virtual void OnBeginAttributes() = 0;
        virtual void OnEndAttributes() = 0;

        virtual void OnRaw(const TStringBuf& yson, EYsonType type) = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////

    struct TYsonConsumerBase
       : public virtual IYsonConsumer {
        void OnRaw(const TStringBuf& ysonNode, EYsonType type) override;
    };

    ////////////////////////////////////////////////////////////////////////////////

}
