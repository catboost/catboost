#pragma once


template <class TPtr>
bool ArePointeesEqual(const TPtr& lhs, const TPtr& rhs) {
    if (!lhs) {
        return !rhs;
    }
    return *lhs == *rhs; // both non-nullptr
}
