// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include "embree_utils/geometry.hpp"

// A very simple material description.
struct Material {
  enum class Type {
    Diffuse, Specular, Refractive
  };

  Material()
  :
    albedo(0.f, 0.f, 0.f),
    ior(1.52f),
    emission(0.f, 0.f, 0.f),
    type(Material::Type::Diffuse),
    emissive(false) {}

  Material(const embree_utils::Vec3fa& a,
           const embree_utils::Vec3fa& e, Material::Type t)
  :
    albedo(a), ior(1.52f), emission(e), type(t),
    emissive(emission.isNonZero())
  {}

  embree_utils::Vec3fa albedo;
  float ior; // index of refraction
  embree_utils::Vec3fa emission;
  Type type;
  bool emissive;
};
