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

struct __attribute__((packed, aligned(alignof(std::uint16_t))))
Triangle {
  Triangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) : v0(a), v1(b), v2(c) {}
  std::uint16_t v0, v1, v2;
};

struct RayShearParams {
  RayShearParams(const embree_utils::Ray& ray);

  embree_utils::Vec3fa o; // offset
  embree_utils::Vec3fa dir; // permuted direction
  std::uint32_t ix, iy, iz; // permutation
  float sx, sy, sz;  // shear
};

struct Sphere : Primitive {
  // This is a hack to force IPU size/align to be compatible with the host:
  struct __attribute__ ((aligned (8))) {
    float x, y, z;
  };
  const float radius;
  const float radius2;

  Sphere(const embree_utils::Vec3fa& c, float r) : x(c.x), y(c.y), z(c.z), radius(r), radius2(r*r) {}
  ~Sphere() {}

  Intersection intersect(std::uint32_t primID, const embree_utils::Ray& ray) const override;

  embree_utils::Vec3fa normal(const Intersection&, const embree_utils::Vec3fa& point) const override {
    return (point - embree_utils::Vec3fa(x, y, z)).normalized();
  }

  embree_utils::Bounds3d getBoundingBox() const override {
    const embree_utils::Vec3fa centre(x, y, z);
    return embree_utils::Bounds3d{centre - radius, centre + radius};
  }
};

struct Disc : Primitive {
  struct __attribute__ ((aligned (8))) {
    float nx, ny, nz;
  };
  float r;
  struct __attribute__ ((aligned (8))) {
    float cx, cy, cz;
  };
  float r2;

  Disc(const embree_utils::Vec3fa& normal, const embree_utils::Vec3fa& centre, float radius)
    : nx(normal.x), ny(normal.y), nz(normal.z), cx(centre.x), cy(centre.y), cz(centre.z), r(radius), r2(radius*radius) {}
  ~Disc() {}

  embree_utils::Vec3fa normal(const Intersection&, const embree_utils::Vec3fa&) const override { return embree_utils::Vec3fa(nx, ny, nz); }

  Intersection intersect(std::uint32_t primID, const embree_utils::Ray& ray) const override;

  embree_utils::Bounds3d getBoundingBox() const override {
    // Disc is always within its bounding sphere (very slack bound is computed in constructor):
    const embree_utils::Vec3fa c(cx, cy, cz);
    return embree_utils::Bounds3d(c - r, c + r);
  }
};
