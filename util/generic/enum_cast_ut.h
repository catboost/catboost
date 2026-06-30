#pragma once

enum class EIntEnum: int {
    Zero = 0,
    One = 1,
    Two = 2
};

enum class EUcharEnum: unsigned char {
    Zero = 0,
    One = 1,
    Two = 2
};

enum class ECharEnum: signed char {
    Zero = 0,
    MinusOne = -1,
    MinusTwo = -2
};

enum class EBoolEnum: bool {
    False = false,
    True = true
};

enum EUnscopedIntEnum {
    UIE_TWO = 2,
};
