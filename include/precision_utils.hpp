// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains utility functions that assist with numerical
// precision improvements in ray-tracing code.

#pragma once

#include <cstdint>
#include <limits>
#include <memory>

#ifdef __IPU__
#include <poplar/HalfFloat.hpp>
#else
#include <Eigen/Dense>
using half = Eigen::half;
#endif

static constexpr float machineEpsilon = std::numeric_limits<float>::epsilon() * .5f;

static constexpr float gamma(int i) {
  const auto ni = machineEpsilon * i;
  return ni / (1 - ni);
}

static constexpr float rayEpsilon = machineEpsilon * 1500.f;

inline
half nextHalfUp(half h) {
  static_assert(sizeof(std::uint16_t) == sizeof(half), "Eigen half badly sized.");
  std::uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  bits += 1;
  half result;
  std::memcpy(&result, &bits, sizeof(bits));
  return result;
}

inline
half roundToHalfNotSmaller(float f) {
  half h = (half)f;
  float ff = (float)h;
  if (ff < f) {
    h = nextHalfUp(h);
  }
  return h;
}
