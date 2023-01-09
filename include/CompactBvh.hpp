// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// CompactBvh is a static array of BVH nodes that references other nodes
// in the same array using relative indices. This is an efficient in memory
// BVH structure for use on IPU. Similar structures are typically common even
// in non-IPU implementations as they are memory efficient and can be moved around
// (e.g. paged) as they do not contain any absolute indices or pointers.

#pragma once

#include "embree_utils/geometry.hpp"
#include "Node.hpp"
#include "Intersection.hpp"
#include "Arrays.hpp"

struct CompactBvh {

  embree_utils::Bounds3d getBoundingBox() const {
    const auto& root = nodes.front();
    return embree_utils::Bounds3d(
      embree_utils::Vec3fa(root.min_x, root.min_y, root.min_z),
      embree_utils::Vec3fa(root.max_x, root.max_y, root.max_z)
    );
  }

  ConstArrayRef<CompactBVH2Node> getNodes() const { return nodes; }
  const std::uint32_t getMaxDepth() const { return stackSize; }

  CompactBvh(ConstArrayRef<CompactBVH2Node> nodesRef,
             std::uint32_t maxTraversalDepth)
  : nodes(nodesRef), stackSize(maxTraversalDepth) {}

  // Return true if the ray intersects any primitive in the BVH. For occlusion
  // tests this is more efficient than calling interesect because it can return early if any
  // primitive intersection is found. This shares the traversal stack with intersect() so can
  // not be called in a parallel thread to other intersection tests.
  template <class Lookup>
  bool occluded(const embree_utils::Ray& ray,
                Lookup& primLookup,
                std::uint32_t excludeGeomID = embree_utils::HitRecord::InvalidGeomID,
                std::uint32_t excludePrimID = embree_utils::HitRecord::InvalidPrimID) const {
    // Push root node onto stack:
    std::uint32_t stack[stackSize];
    auto ref = ArrayRef(stack, stackSize);
    WrappedArray<std::uint32_t> toVisit(ref);
    toVisit.push_back(0);

    // Setup intersection test:
    const embree_utils::Vec3fa invRayDir(
      1.f / ray.direction.x,
      1.f / ray.direction.y,
      1.f / ray.direction.z
    );

    while (!toVisit.empty()) {
      auto currentIndex = toVisit.back();
      toVisit.pop_back();
      const CompactBVH2Node& node = nodes[currentIndex];

      // check if ray hits bounds by testing against each axis aligned slab:
      float t0 = ray.tMin;
      float t1 = ray.tMax;

      if (node.intersect(ray.origin, invRayDir, t0, t1)) {
        if (node.geomID != CompactBVH2Node::InvalidGeomID) {
          // Node is a leaf so we must intersect the geometry:
          auto* prim = primLookup(node.geomID, node.primID);
          auto intersection = prim->intersect(node.primID, ray);
          if (intersection.t > ray.tMin && intersection.t < ray.tMax) {
            return true; // Early exit on first hit
          }
        } else {
          // Node is interior so push the two children onto the
          // stack (First child is always next in the array):
          toVisit.push_back(node.secondChildIndex);
          toVisit.push_back(currentIndex + 1);
        }
      }
    }

    return false;
  }

  template <class Lookup>
  Intersection intersect(
    const embree_utils::Ray& ray,
    Lookup& primLookup
  ) const {
    // Push root node onto stack:
    std::uint32_t stack[stackSize];
    auto ref = ArrayRef(stack, stackSize);
    WrappedArray<std::uint32_t> toVisit(ref);
    toVisit.push_back(0);

    // Setup intersection test:
    const embree_utils::Vec3fa invRayDir(
      1.f / ray.direction.x,
      1.f / ray.direction.y,
      1.f / ray.direction.z
    );

    auto closestIntersection = Intersection(ray.tMax, nullptr);

    while (!toVisit.empty()) {
      auto currentIndex = toVisit.back();
      toVisit.pop_back();
      const CompactBVH2Node& node = nodes[currentIndex];

      // check if ray hits bounds by testing against each axis aligned slab:
      float t0 = ray.tMin;
      float t1 = closestIntersection.t;

      if (node.intersect(ray.origin, invRayDir, t0, t1)) {
        if (node.geomID != CompactBVH2Node::InvalidGeomID) {
          // Debug: visualise Bounds of leaf nodes:
          // if (t0 < closestIntersection.t) {
          //   closestIntersection.t = t0;
          //   closestIntersection.geomID = node.geomID;
          //   closestIntersection.primID = node.primID;
          // }

          // Node is a leaf so intersect the geometry:
          const Primitive* prim = primLookup(node.geomID, node.primID);
          auto intersection = prim->intersect(node.primID, ray);
          intersection.prim = prim;

          // If this is closest intersection so far save it:
          if (intersection.t > ray.tMin && intersection.t < closestIntersection.t) {
            intersection.geomID = node.geomID;
            // intersection.primID is set by the intersect method.
            closestIntersection = intersection;
          }
        } else {
          // Node is interior so push the two children onto the
          // stack (First child is always next in the array):
          toVisit.push_back(node.secondChildIndex);
          toVisit.push_back(currentIndex + 1);
        }
      }
    }

    return closestIntersection;
  }

private:
  const ConstArrayRef<CompactBVH2Node> nodes;
  const std::uint32_t stackSize;
};
