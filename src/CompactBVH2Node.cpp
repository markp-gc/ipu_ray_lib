// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <CompactBVH2Node.hpp>

bool CompactBVH2Node::intersect(const embree_utils::Vec3fa& o, const embree_utils::Vec3fa& i, float& t0, float& t1) const {
  // Test ray against each axis aligned slab in turn:
  if (intersectRaySlab(i.x, o.x, min_x, max_x, t0, t1)) {
    if (intersectRaySlab(i.y, o.y, min_y, max_y, t0, t1)) {
      if (intersectRaySlab(i.z, o.z, min_z, max_z, t0, t1)) {
        return true;
      }
    }
  }

  return false;
}
