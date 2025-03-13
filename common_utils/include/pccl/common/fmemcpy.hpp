#pragma once

#include <cstddef>

#ifndef _MSC_VER
#define RESTRICT __restrict__
#else
#define RESTRICT __restrict
#endif

void fast_memcpy(void *RESTRICT dst,
                           const void *RESTRICT src,
                           size_t length);
