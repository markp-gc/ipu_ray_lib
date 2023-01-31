// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

// Debug macro to print from IPU vertex code.

#pragma once

#if DEBUG
#include <print.h>
#define DEBUG_PRINT(fmt, args...) printf(fmt, args)
#else
#define DEBUG_PRINT(fmt, args...)
#endif
