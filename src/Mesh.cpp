// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
// Copyright (c) 1998-2015, Matt Pharr, Greg Humphreys, and Wenzel Jakob. All rights reserved.

#include <Mesh.hpp>

template <template<class T> class Storage>
TriangleIntersection
TriangleMesh<Storage>::intersectTriangle(std::uint32_t index, const RayShearParams& transform, const float tFar) const {
  // Get the triangles vertices:
  const auto& tri = triangles[index];
  const auto& p0 = vertices[tri.v0];
  const auto& p1 = vertices[tri.v1];
  const auto& p2 = vertices[tri.v2];

  // Translate vertices into ray coordinate system:
  auto p0t = p0 - transform.o;
  auto p1t = p1 - transform.o;
  auto p2t = p2 - transform.o;

  // Permute coordinate system so largest component is z:
  p0t = p0t.permute(transform.ix, transform.iy, transform.iz);
  p1t = p1t.permute(transform.ix, transform.iy, transform.iz);
  p2t = p2t.permute(transform.ix, transform.iy, transform.iz);

  // Shear permuted triangle z axis to align with ray z axis:
  p0t.x += transform.sx * p0t.z;
  p0t.y += transform.sy * p0t.z;
  p1t.x += transform.sx * p1t.z;
  p1t.y += transform.sy * p1t.z;
  p2t.x += transform.sx * p2t.z;
  p2t.y += transform.sy * p2t.z;

  // Compute edge coefficients:
  auto e0 = p1t.x * p2t.y - p1t.y * p2t.x;
  auto e1 = p2t.x * p0t.y - p2t.y * p0t.x;
  auto e2 = p0t.x * p1t.y - p0t.y * p1t.x;

  // Perform triangle edge and determinant tests
  if ((e0 < 0 || e1 < 0 || e2 < 0) &&
      (e0 > 0 || e1 > 0 || e2 > 0)) {
    return TriangleIntersection{0.f, 0.f, 0.f, 0.f};
  }

  auto det = e0 + e1 + e2;
  if (det == 0) {
    return TriangleIntersection{0.f, 0.f, 0.f, 0.f};
  }

  // Compute scaled hit distance to triangle and test against ray $t$ range
  p0t.z *= transform.sz;
  p1t.z *= transform.sz;
  p2t.z *= transform.sz;
  auto tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
  if (det < 0.f && (tScaled >= 0.f || tScaled < tFar * det)) {
    return TriangleIntersection{0.f, 0.f, 0.f, 0.f};
  } else if (det > 0.f && (tScaled <= 0.f || tScaled > tFar * det)) {
    return TriangleIntersection{0.f, 0.f, 0.f, 0.f};
  }

  // Compute barycentric coordinates and $t$ value for triangle intersection
  auto invDet = 1 / det;
  auto b0 = e0 * invDet;
  auto b1 = e1 * invDet;
  auto b2 = e2 * invDet;
  auto t = tScaled * invDet;

  // Manage rounding error to guarantee tmax is greater than zero:

  // Compute worst error for tmax:
  auto maxZt = embree_utils::Vec3fa(p0t.z, p1t.z, p2t.z).abs().maxc();
  auto deltaZ = gamma(3) * maxZt;

  auto maxXt = embree_utils::Vec3fa(p0t.x, p1t.x, p2t.x).abs().maxc();
  auto maxYt = embree_utils::Vec3fa(p0t.y, p1t.y, p2t.y).abs().maxc();
  auto deltaX = gamma(5) * (maxXt + maxZt);
  auto deltaY = gamma(5) * (maxYt + maxZt);

  auto deltaE =
      2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

  // Final check on tmax:
  auto maxE = embree_utils::Vec3fa(e0, e1, e2).abs().maxc();
  auto deltaT = 3 *
                  (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
                  std::abs(invDet);
  if (t <= deltaT) return TriangleIntersection{0.f, b0, b1, b2};

  return TriangleIntersection{t, b0, b1, b2};
}

template struct TriangleMesh<ConstArrayRef>;

#ifndef __IPU__
template struct TriangleMesh<std::vector>;
#endif
