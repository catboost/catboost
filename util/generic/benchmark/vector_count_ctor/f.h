#pragma once

#include <cstddef>

// functions are declared in a separate translation unit so that compiler won't be able to see the
// value of `size` during compilation.

void CreateYvector(const size_t size, const size_t count);
void CreateCarray(const size_t size, const size_t count);
