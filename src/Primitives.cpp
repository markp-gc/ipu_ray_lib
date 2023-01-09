// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <Primitives.hpp>

RayShearParams::RayShearParams(const embree_utils::Ray& ray)
  : o(ray.origin), dir(ray.direction)
{
  // Calc indices to permute coordinate system so largest component is z:
  iz = dir.maxi();
  ix = iz + 1;
  if (ix == 3) { ix = 0; }
  iy = ix + 1;
  if (iy == 3) { iy = 0; }

  // Store permuted ray direction:
  dir = dir.permute(ix, iy, iz);

  // Compute shear that aligns triangles to permuted z axis:
  sx = -dir.x / dir.z;
  sy = -dir.y / dir.z;
  sz = 1.f / dir.z;
}

Intersection Sphere::intersect(std::uint32_t primID, const embree_utils::Ray& ray) const {
  embree_utils::Vec3fa f = centre - ray.origin;
  auto rd2 = 1.f / ray.direction.squaredNorm();
  auto tca = f.dot(ray.direction) * rd2;
  if (tca < 0.f) { return Intersection::Failed(); }
  embree_utils::Vec3fa l = f - ray.direction * tca;
  auto l2 = l.squaredNorm();
  if (l2 > radius2) { return Intersection::Failed(); }
  auto td = sqrtf(radius2 - l2) * rd2;
  auto t0 = tca - td;
  auto t1 = tca + td;

  if (t0 > t1) { std::swap(t0, t1); }
  if (t0 < ray.tMin) {
      t0 = t1;
      if (t0 < ray.tMin) { return Intersection::Failed(); }
  }

  Intersection result(t0, this);
  result.primID = 0;
  return result;
}

Intersection Disc::intersect(std::uint32_t primID, const embree_utils::Ray& ray) const {
  auto angle = n.dot(ray.direction);
  if (angle != 0.f) {
    auto d = std::abs(c.dot(n));
    auto t = -((n.dot(ray.origin)) + d) / angle;
    if (t > machineEpsilon) {
      const auto hitPoint = ray.origin + ray.direction * t;
      auto d2 = (hitPoint - c).squaredNorm();
      if (d2 < r2) {
        Intersection result(t, this);
        result.primID = 0;
        return result;
      }
    }
  }

  return Intersection::Failed();
}
