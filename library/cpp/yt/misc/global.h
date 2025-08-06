#pragma once

//! Defines a global variable that is initialized on its first access.
/*!
 *  In contrast to a usual variable with static storage duration, this one
 *  is not susceptible to initialization order fiasco issues.
 */
#define YT_DEFINE_GLOBAL(type, name, ...) \
    inline type& name() \
    { \
        static type result{__VA_ARGS__}; \
        return result;  \
    }
