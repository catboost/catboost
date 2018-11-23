#pragma once

namespace NCB {

    enum class EObjectsOrder {
        Ordered, // order is important
        RandomShuffled,
        Undefined
    };

    EObjectsOrder Combine(EObjectsOrder srcOrder, EObjectsOrder subsetOrder);

}
