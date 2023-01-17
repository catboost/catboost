#include "order.h"

#include <catboost/libs/helpers/exception.h>

#include <util/system/yassert.h>

namespace NCB {

    EObjectsOrder Combine(EObjectsOrder srcOrder, EObjectsOrder subsetOrder) {
        switch (srcOrder) {
            case EObjectsOrder::Ordered:
                return subsetOrder; // TODO(akhropov). Prohibit shuffling ordered data?

            case EObjectsOrder::RandomShuffled:
                return EObjectsOrder::RandomShuffled;

            case EObjectsOrder::Undefined:
                switch (subsetOrder) {
                    case EObjectsOrder::RandomShuffled:
                        return EObjectsOrder::RandomShuffled;
                    case EObjectsOrder::Ordered:
                    case EObjectsOrder::Undefined:
                        return EObjectsOrder::Undefined;
                }
        }
        CB_ENSURE(false, "This place can't be reached");
        return EObjectsOrder::Undefined; // to make compiler happy
    }

}
