// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains our own BVH node data structures to
// build a custom BVH using Embree as a backend.

#pragma once

#include <embree3/rtcore.h>
#include <embree3/rtcore_common.h>
#include <embree3/rtcore_builder.h>

#include <emmintrin.h>

#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <vector>
#include <functional>
#include <atomic>

#include "geometry.hpp"

namespace embree_utils {

struct Node {
  virtual float sah() = 0;
};

struct InnerNode : public Node {
  Bounds3d bounds;
  Node* children[2];

  InnerNode() {
    children[0] = children[1] = nullptr;
  }

  float sah() override {
    throw std::runtime_error("Does this get called?");
    return 2.f;
  }

  static void* create(RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr) {
    if (numChildren != 2) { throw std::runtime_error("Expected exactly 2 children"); }
    void* ptr = rtcThreadLocalAlloc(alloc, sizeof(InnerNode), 16);
    return (void*) new (ptr) InnerNode;
  }

  static void setChildren(void* nodePtr, void** childPtr, unsigned int numChildren, void* userPtr) {
    if (numChildren != 2) { throw std::runtime_error("Expected exactly 2 children"); }
    for (size_t i = 0; i < 2; ++i) {
      ((InnerNode*)nodePtr)->children[i] = (Node*)childPtr[i];
    }
  }

  static void setBounds(void* nodePtr, const RTCBounds** bounds, unsigned int numChildren, void* userPtr) {
    if (numChildren != 2) { throw std::runtime_error("Expected exactly 2 children"); }

    // Compute union of children's bounding boxes:
    Bounds3d& nodeBounds = ((InnerNode*)nodePtr)->bounds;
    for (size_t i = 0; i < 2; ++i) {
      nodeBounds +=
        Bounds3d(
          Vec3fa(bounds[i]->lower_x, bounds[i]->lower_y, bounds[i]->lower_z),
          Vec3fa(bounds[i]->upper_x, bounds[i]->upper_y, bounds[i]->upper_z)
        );
    }
  }
};

struct LeafNode : public Node {
  Bounds3d bounds;
  unsigned primID;
  unsigned geomID;

  LeafNode (const Bounds3d& bounds, unsigned pId, unsigned gId)
    : primID(pId), geomID(gId), bounds(bounds) {}

  float sah() override {
    throw std::runtime_error("Does this get called?");
    return 1.0f;
  }

  static void* create(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr) {
    if (numPrims != 1) { throw std::runtime_error("Expected exactly 1 primitive"); }
    Bounds3d bounds(
      Vec3fa(prims[0].lower_x, prims[0].lower_y, prims[0].lower_z),
      Vec3fa(prims[0].upper_x, prims[0].upper_y, prims[0].upper_z)
    );
    void* ptr = rtcThreadLocalAlloc(alloc, sizeof(LeafNode), 16);
    return (void*) new (ptr) LeafNode(bounds, prims[0].primID, prims[0].geomID);
  }
};

} // end namespace embree_utils
