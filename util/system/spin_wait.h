#pragma once

struct TSpinWait {
    TSpinWait() noexcept;

    void Sleep() noexcept;

    unsigned T;
    unsigned C;
};
