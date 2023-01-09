// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains std::ostream operator overloads for some Embree and embree_utils types.

#pragma once

#include "geometry.hpp"
#include "bvh.hpp"

std::ostream& operator << (std::ostream& os, const RTCHit& hit) {
  os << "hit prim: " << hit.primID << " hit geom: " << hit.geomID << " ";
  os << "hit normal: " << hit.Ng_x << " " << hit.Ng_x << " " << hit.Ng_x << " ";
  os << "hit u,v: " << hit.u << " " << hit.v;
  return os;
}

std::ostream& operator << (std::ostream& os, const RTCRay& ray) {
  os << "ray origin: " << ray.org_x << " " << ray.org_y << " " << ray.org_z << " ";
  os << "dir: " << ray.dir_x << " " << ray.dir_y << " " << ray.dir_z << " ";
  os << "near: " << ray.tnear << " far: " << ray.tfar;
  return os;
}

std::ostream& operator << (std::ostream& os, const embree_utils::Vec3fa& v) {
  os << v.x << " " << v.y << " " << v.z;
  return os;
}

std::ostream& operator << (std::ostream& os, const embree_utils::Bounds3d& b) {
  os << b.min << " -> " << b.max;
  return os;
}
