// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// Functionality to convert from custom BVH built with Embree
// into a more compact and efficient data structure.

#include "embree_utils/node.hpp"
#include "Node.hpp"

/// Recursively traverse a BVH2 sub-tree and flatten it into an array of compact nodes.
std::uint32_t flattenBVH2Tree(const embree_utils::Node* node, std::vector<CompactBVH2Node>& compactTree, std::uint32_t depth, std::uint32_t& index, std::uint32_t& maxDepth);

// Convert a custom BVH built using Embree as a backend
// into a compact array of nodes giving a more efficient data structure.
// The returned array can be wrapped into a CompactBvh to instansiate an
// efficient BVH object.
static std::vector<CompactBVH2Node> buildCompactBvh(
    const embree_utils::Node* startNode,
    std::uint32_t maxNodes,
    std::uint32_t& maxDepth)
{
  std::vector<CompactBVH2Node> nodes(maxNodes);  
  std::uint32_t indexTracker = 0;
  maxDepth = 0;
  flattenBVH2Tree(startNode, nodes, 1, indexTracker, maxDepth);
  return nodes;
}
