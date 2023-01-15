// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// Misc. rendering utility functions.

#pragma once

#include <Arrays.hpp>
#include <Primitives.hpp>
#include <CompactBvh.hpp>
#include <Material.hpp>

/// Update the hit record with intersection info, advance ray to hit
/// point, and return a pointer to the primitive that was hit.
inline
void updateHit(const Intersection& i, embree_utils::HitRecord& hit) {
  // Mark a geometry hit the same way embree does:
  hit.geomID = i.geomID;
  hit.primID = i.primID;
  hit.r.tMax = i.t;
  hit.r.origin += hit.r.direction * i.t; // Update ray origin with the hit point
  hit.normal = i.prim->normal(i, hit.r.origin);
}

/// Templated on the callback that can lookup a Primitive
/// from its geom and prim IDs.
template <class T>
void traceShadowRay(const CompactBvh& bvh,
                    const ConstArrayRef<std::uint32_t>& matIDs,
                    const ConstArrayRef<Material>& materials,
                    float shadowRayOffset,
                    float ambient,
                    embree_utils::TraceResult& result,
                    T& primLookupFunc,
                    const embree_utils::Vec3fa& lightWorldPos) {
  auto& hit = result.h;
  auto intersected = bvh.intersect(hit.r, primLookupFunc);
  if (intersected) {
    updateHit(intersected, hit);
    const auto& material = materials[matIDs[hit.geomID]];

    // For rays that hit cast a shadow ray to an arbitrary point
    // to test BVH occlusion function:
    auto shadowRay = hit.r;
    auto lightOffset = lightWorldPos - shadowRay.origin;
    shadowRay.direction = lightOffset.normalized();
    shadowRay.tMax = std::sqrt(lightOffset.squaredNorm());

    shadowRay.tMin = shadowRayOffset;
    auto matRgb = material.albedo;
    auto color = matRgb * ambient;
    // Occlusion test is more efficient than intersect for casting shadow rays:
    if (!bvh.occluded(shadowRay, primLookupFunc)) {
      color += matRgb * shadowRay.direction.dot(hit.normal); // lambertian
    }
    result.rgb = color;
  }
}

inline
embree_utils::Vec3fa pixelToRayDir(float x, float y,
                                   float w, float h,
                                   float tanTheta) { // tanTheta = tan(fov/2)
  const float aspect = w / h;
  x = (x / w) - .5f;
  y = (y / h) - .5f;
  return embree_utils::Vec3fa(
        2.f * x * aspect * tanTheta,
        -2.f * y * tanTheta,
        -1.f).normalized();
}
