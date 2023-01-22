// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains a few fundamental Primitive classes. (Note: mesh primitives are found elsewhere).

#pragma once

#include "Intersection.hpp"

struct Primitive {
  virtual Intersection intersect(std::uint32_t primID, const embree_utils::Ray& ray) const {
    return Intersection::Failed();
  }

  virtual embree_utils::Vec3fa normal(const Intersection& result, const embree_utils::Vec3fa& hitPoint) const {
    return embree_utils::Vec3fa();
  }

  virtual embree_utils::Bounds3d getBoundingBox() const { return embree_utils::Bounds3d(); }
};

struct __attribute__((packed, aligned(alignof(std::uint32_t))))
Triangle {
  std::uint32_t v0, v1, v2;
};

struct RayShearParams {
  RayShearParams(const embree_utils::Ray& ray);

  embree_utils::Vec3fa o; // offset
  embree_utils::Vec3fa dir; // permuted direction
  std::uint32_t ix, iy, iz; // permutation
  float sx, sy, sz;  // shear
};

struct Sphere : Primitive {
  embree_utils::Vec3fa centre;
  const float radius;
  const float radius2;

  Sphere(const embree_utils::Vec3fa& c, float r) : centre(c), radius(r), radius2(r*r) {}
  ~Sphere() {}

  Intersection intersect(std::uint32_t primID, const embree_utils::Ray& ray) const override;

  embree_utils::Vec3fa normal(const Intersection&, const embree_utils::Vec3fa& point) const override {
    return (point - centre).normalized();
  }

  embree_utils::Bounds3d getBoundingBox() const override {
    return embree_utils::Bounds3d{centre - radius, centre + radius};
  }
};

struct Disc : Primitive {
  embree_utils::Vec3fa n;
  float r;
  embree_utils::Vec3fa c;
  float r2;

  Disc(const embree_utils::Vec3fa& normal, const embree_utils::Vec3fa& centre, float radius)
    : n(normal.normalized()), c(centre), r(radius), r2(radius*radius) {}
  ~Disc() {}

  embree_utils::Vec3fa normal(const Intersection&, const embree_utils::Vec3fa&) const override { return n; }

  Intersection intersect(std::uint32_t primID, const embree_utils::Ray& ray) const override;

  embree_utils::Bounds3d getBoundingBox() const override {
    // Disc is always within its bounding sphere (very slack bound is computed in constructor):
    return embree_utils::Bounds3d(c - r, c + r);
  }
};
