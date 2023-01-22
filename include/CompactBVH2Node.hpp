// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains a compact BVH node representation and functionality.

#pragma once

#ifdef __IPU__
#include <ipu_vector_math>
#endif

#include "precision_utils.hpp"
#include "embree_utils/geometry.hpp"

inline
bool intersectRaySlab(float componentInvDir, float componentOrigin, float slabMin, float slabMax, float& t0, float& t1) {

#ifdef __IPU__
  // Use some intrinsics to help IPU compiler with vectorisation:
  float2 tmin_max = {slabMin, slabMax};
  tmin_max -= float2{componentOrigin, componentOrigin};
  tmin_max *= float2{componentInvDir, componentInvDir};

  if (tmin_max[0] > tmin_max[1]) {
    auto tmp = tmin_max[1];
    tmin_max[1] = tmin_max[0];
    tmin_max[0] = tmp;
  }

  // Make sure we never miss a potential hit due to rounding error:
  tmin_max[1] *= 1 + 2 * gamma(3);

  t0 = tmin_max[0] > t0 ? tmin_max[0] : t0;
  t1 = tmin_max[1] < t1 ? tmin_max[1] : t1;
#else

  float tmin = (slabMin - componentOrigin) * componentInvDir;
  float tmax = (slabMax - componentOrigin) * componentInvDir;
  if (tmin > tmax) { std::swap(tmin, tmax); }

  // Make sure we never miss a potential hit due to rounding error:
  tmax *= 1 + 2 * gamma(3);

  t0 = tmin > t0 ? tmin : t0;
  t1 = tmax < t1 ? tmax : t1;

#endif

  if (t0 > t1) return false;
  return true;
}

struct
__attribute__ ((aligned (8)))
CompactBVH2Node {
  static constexpr auto InvalidGeomID = std::numeric_limits<std::uint16_t>::max();
  static constexpr auto InvalidPrimID = std::numeric_limits<std::uint32_t>::max();

  // Explicitly list the bounds element by element to get a compact structure:
  float min_x, min_y, min_z;

  // If this is a leaf we store the primitive IDs, otherwise we store
  // the index of the second child node. (The first child node is always
  // the next node in the array).
  union {
    std::uint32_t primID;
    std::uint32_t secondChildIndex;
  };

  // Store width, height, and depth of bounding box at half precision.
  // This saves 25% BVH node memory on device:
  half dx, dy, dz;

  std::uint16_t geomID; // If this == InvalidGeomID then the node is an inner node.

  /// Test if a ray intersects this node's bounding box and update
  /// t0 and t1 to the new possible intersection ranges:
  bool intersect(const embree_utils::Vec3fa& o, const embree_utils::Vec3fa& i, float& t0, float& t1) const;

  embree_utils::Bounds3d toBounds() const {
    return embree_utils::Bounds3d(
      embree_utils::Vec3fa(min_x, min_y, min_z),
      embree_utils::Vec3fa(min_x + (float)dx, min_y + (float)dy, min_z + (float)dz)
    );
  }
};
