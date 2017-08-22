#include "messagext_lite.h"

#include "messagext.h"

#include <util/generic/yexception.h>


namespace NProtoBufInternal {

    IOutputStream& operator <<(IOutputStream& output, const NProtoBufInternal::TAsBinary& wrappedMessage) {
        bool success = wrappedMessage.Message_.SerializeToStream(&output);
        if (Y_UNLIKELY(!success)) {
            ythrow yexception() << "Cannot serialize a protobuf with AsBinary() (required fields missing?)";
        }
        return output;
    }


    IOutputStream& operator <<(IOutputStream& output, const NProtoBufInternal::TAsStreamSeq& wrappedMessage) {
        ::Save(&output, wrappedMessage.Message_);
        return output;
    }

}
