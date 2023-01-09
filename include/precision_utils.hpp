// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains utility functions that assist with numerical
// precision improvements in ray-tracing code.

#pragma once

#include <cstdint>
#include <limits>

static constexpr float machineEpsilon = std::numeric_limits<float>::epsilon() * .5f;

static constexpr float gamma(int i) {
  const auto ni = machineEpsilon * i;
  return ni / (1 - ni);
}

