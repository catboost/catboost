#pragma once

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! A callable that does nothing.
/*!
 *  Serves as a default template argument where a lambda cannot: each spelling of a
 *  default argument instantiates its own closure type, so a declaration and its
 *  definition would disagree on the enclosing type.
 */
struct TNoOp
{
    void operator()() const
    { }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
