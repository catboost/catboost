#pragma once

#include "stub/stl.h"

namespace Pire {
    class BudgetExceeded: public Error {
    public:
        BudgetExceeded(): Error("Pire compilation operation budget exceeded") {}
    };

    namespace Impl {
        inline thread_local size_t* OperationBudget = nullptr;

        // Charge before doing work; division avoids overflow in bulk charges.
        inline void ChargeOperations(size_t count, size_t repetitions = 1) {
            if (OperationBudget && count) {
                if (repetitions > *OperationBudget / count) {
                    *OperationBudget = 0;
                    throw BudgetExceeded();
                }
                *OperationBudget -= count * repetitions;
            }
        }

        // Internal singleton initialization must not spend a caller's budget.
        class ScopedOperationBudgetPause {
        public:
            ScopedOperationBudgetPause() noexcept
                : Previous_(OperationBudget)
            {
                OperationBudget = nullptr;
            }

            ~ScopedOperationBudgetPause() { OperationBudget = Previous_; }

            ScopedOperationBudgetPause(const ScopedOperationBudgetPause&) = delete;
            ScopedOperationBudgetPause& operator=(const ScopedOperationBudgetPause&) = delete;

        private:
            size_t* Previous_;
        };
    }

    // Applies to compilation on this thread. Nested scopes restore the previous
    // remaining budget. Destroy on the creating thread, in reverse scope order.
    // Units count algorithmic work, not CPU instructions, bytes or elapsed time.
    // Linear input preparation and plain FSM copies are free; see README.md.
    class ScopedOperationBudget {
    public:
        explicit ScopedOperationBudget(size_t operations) noexcept
            : Remaining_(operations)
            , Previous_(Impl::OperationBudget)
        {
            Impl::OperationBudget = &Remaining_;
        }

        ~ScopedOperationBudget() {
            Impl::OperationBudget = Previous_;
        }

        size_t Remaining() const noexcept { return Remaining_; }

        ScopedOperationBudget(const ScopedOperationBudget&) = delete;
        ScopedOperationBudget& operator=(const ScopedOperationBudget&) = delete;

    private:
        size_t Remaining_;
        size_t* Previous_;
    };
}
