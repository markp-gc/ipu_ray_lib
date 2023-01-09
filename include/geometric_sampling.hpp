// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains fundamental geometric sampling routines.

#include <math/sincos.hpp>

// Given a uniform sample on the unit square return a uniform sample on the unit disc.
inline std::pair<float, float> sampleDiscConcentric(float u1, float u2) {
  float ux = 2.f * u1 - 1.f;
  float uy = 2.f * u2 - 1.f;

  if (ux == 0.f && uy == 0.f) {
    return {ux, uy};
  }

  float r;
  float th;
  if (std::abs(ux) > std::abs(uy)) {
    r = ux;
    th = embree_utils::Piby4 * (uy / ux);
  } else {
    r = uy;
    th = embree_utils::Piby2 - embree_utils::Piby4 * (ux / uy);
  }

  float s, c;
  sincos(th, s, c);
  return {r * c, r * s};
}

// Given a uniform sample on the unit square return a uniform sample on the unit hemisphere.
inline embree_utils::Vec3fa sampleHemisphere(float u1, float u2) {
  const float r = std::sqrt(1.f - u1 * u1);
  const float phi = 2.f * embree_utils::Pi * u2;
  float s, c;
  sincos(phi, s, c);
  return embree_utils::Vec3fa(c * r, s * r, u1);
}

// Given a uniform sample on the unit square return a cosine distributed sample on the unit hemisphere.
inline embree_utils::Vec3fa cosineSampleHemisphere(float u1, float u2) {
  const auto [x, y] = sampleDiscConcentric(u1, u2);
  float z = std::sqrt(std::max(0.f, 1.f - x*x - y*y));
  return embree_utils::Vec3fa(x, y, z);
}

// Return the PDF for a cosine hemisphere sample.
inline float cosineHemispherePdf(const embree_utils::Vec3fa& incomingRayDirInTangentSpace) {
  const float cosTh = incomingRayDirInTangentSpace.z;
  return embree_utils::InvPi * cosTh;
}

// Given a uniform random number and current throughput return whether
// to terminate the path having re-weighted the throughput to account
// for the loss of energy due to random stopping:
inline bool evaluateRoulette(float u1, embree_utils::Vec3fa& throughput) {
  const float p = throughput.maxc();
  if (p == 0.f || u1 > p) {
    return true; // caller should stop path tracing
  }
  throughput *= 1.f / p;
  return false; // caller should continue, throughput already re-weighted
}
