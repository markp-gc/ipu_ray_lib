// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains a BvhBuilder object that allows us to build a custom BVH using Embree as a backend.

#pragma once

#include "node.hpp"

#include <exception>
#include <sstream>

namespace embree_utils {

/// RAII wrapper for an Embree BVH object.
struct Bvh {
  Bvh(const RTCDevice& d) : device(d), bvh(rtcNewBVH(d)) {}

  ~Bvh() { rtcReleaseBVH(bvh); }

  const RTCDevice& device;
  RTCBVH bvh;
};

/// BvhBuilder allows us to build a custom BVH tree using
/// Embree as a backend (i.e. use Embree's partitioning
/// algorithm but output our own node tree data-structure).
struct BvhBuilder {
  BvhBuilder(const RTCDevice& device)
    : bvh(device),
      args(rtcDefaultBuildArguments()),
      innerCount(0), leafCount(0),
      rootNode(nullptr)
  {}

  virtual ~BvhBuilder() {}

  void build(std::vector<RTCBuildPrimitive>& buildPrimitives) {

    auto err = rtcGetDeviceError(bvh.device);
    if (err != RTC_ERROR_NONE) {
      std::stringstream ss;
      ss << "Not starting BVH build. RTCDevice is in error state with code: " << err;
      throw std::runtime_error(ss.str());
    }

    // BVH build configuration hard coded for now:
    args.byteSize = sizeof(args);
    args.buildFlags = RTC_BUILD_FLAG_NONE;
    args.buildQuality = RTC_BUILD_QUALITY_MEDIUM;
    args.maxBranchingFactor = 2;
    args.maxDepth = 64;
    args.sahBlockSize = 1;
    args.minLeafSize = 1;
    args.maxLeafSize = 1;
    args.traversalCost = 1.0f;
    args.intersectionCost = 8.0f;
    args.bvh = bvh.bvh;
    args.primitives = buildPrimitives.data();
    args.primitiveCount = buildPrimitives.size();
    args.primitiveArrayCapacity = buildPrimitives.capacity();
    args.createNode = BvhBuilder::createInner;
    args.setNodeChildren = InnerNode::setChildren;
    args.setNodeBounds = InnerNode::setBounds;
    args.createLeaf = BvhBuilder::createLeaf;
    args.splitPrimitive = nullptr;
    args.buildProgress = nullptr;
    args.userPtr = this;

    rootNode = (Node*)rtcBuildBVH(&args);

    if (rootNode == nullptr) {
      err = rtcGetDeviceError(bvh.device);
      std::stringstream ss;
      ss << "Embree rtcBuildDBVH() returned NULL. RTCDevice error code: " << err;
      throw std::logic_error(ss.str());
    }
  }

  const Node* getRoot() const {
    return rootNode;
  }

  std::uint32_t nodeCount() const {
    return leafCount + innerCount;
  }

  using TraversalFunction = std::function<void(const Node*)>;

  void depthFirstTraversal(Node* startNode, const TraversalFunction& visit) {
    std::vector<Node*> stack;
    stack.push_back(startNode);

    while (!stack.empty()) {
      auto current = stack.back();
      stack.pop_back();
      visit(current);

      if (current == nullptr) {
        throw std::runtime_error("Null node encountered during traversal");
      } else if (auto* inner = dynamic_cast<const InnerNode*>(current)) {
        // Reverse order of children to match a recursive traversal order (easier to debug):
        stack.push_back(inner->children[1]);
        stack.push_back(inner->children[0]);
      } else if (auto* leaf = dynamic_cast<const LeafNode*>(current)) {
      } else {
        throw std::runtime_error("Unknown derived node encountered during traversal.");
      }
    }
  }

  Bvh bvh;
  RTCBuildArguments args;
  std::atomic<unsigned> innerCount;
  std::atomic<unsigned> leafCount;
  Node* rootNode;

  static void* createInner(RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr) {
    ((BvhBuilder*)userPtr)->innerCount += 1;
    return InnerNode::create(alloc, numChildren, nullptr);
  }

  static void* createLeaf(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr) {
    ((BvhBuilder*)userPtr)->leafCount += 1;
    return LeafNode::create(alloc, prims, numPrims, userPtr);
  }
};


} // end namespace embree_utils
