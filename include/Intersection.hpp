// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include "precision_utils.hpp"
#include "embree_utils/geometry.hpp"

// fwd declare Primitive
struct Primitive;

// Structure to hold result of a ray intersection with primitive geometry.
struct Intersection {
  static constexpr auto InvalidPrimId = std::numeric_limits<std::uint32_t>::max();

  static Intersection Failed() { return Intersection(); }

  /// Default constructed values => no intersection
  Intersection() : primID(InvalidPrimId), t(0.f), prim(nullptr) {}

  /// Intersection with specified ray param and primitive ID set to 0.
  /// (I.e. Use for single geometric entities).
  Intersection(float _t, const Primitive* hit) : primID(InvalidPrimId), t(_t), prim(hit) {}

  /// Construct setting primID and parametric ray intersection point.
  Intersection(std::uint32_t geom, std::uint32_t prim, float _t) : geomID(geom), primID(prim), t(_t) {}

  operator bool() const { return prim != nullptr; }

  std::uint32_t geomID;
  std::uint32_t primID; // Primitive ID at intersection == InvalidPrimId if no intersection
  float t; // ray parameter at intersection, <= 0.f if no intersection
  const Primitive* prim; // Pointer to the primitive that was hit
  embree_utils::Vec3fa normal;
};
