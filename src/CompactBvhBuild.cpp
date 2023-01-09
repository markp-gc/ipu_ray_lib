// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <CompactBvhBuild.hpp>

std::uint32_t flattenBVH2Tree(const embree_utils::Node* node, std::vector<CompactBVH2Node>& compactTree, std::uint32_t depth, std::uint32_t& index, std::uint32_t& maxDepth) {
  CompactBVH2Node& compactNode = compactTree[index];
  auto myIndex = index;
  index += 1;

  if (depth > maxDepth) {
    maxDepth = depth;
  }

  if (node == nullptr) {
    throw std::runtime_error("Null node encountered during traversal");
  } else if (auto* leaf = dynamic_cast<const embree_utils::LeafNode*>(node)) {
    compactNode.min_x = leaf->bounds.min.x;
    compactNode.min_y = leaf->bounds.min.y;
    compactNode.min_z = leaf->bounds.min.z;
    compactNode.max_x = leaf->bounds.max.x;
    compactNode.max_y = leaf->bounds.max.y;
    compactNode.max_z = leaf->bounds.max.z;
    compactNode.geomID = leaf->geomID;
    compactNode.primID = leaf->primID;
  } else if (auto* inner = dynamic_cast<const embree_utils::InnerNode*>(node)) {
    compactNode.min_x = inner->bounds.min.x;
    compactNode.min_y = inner->bounds.min.y;
    compactNode.min_z = inner->bounds.min.z;
    compactNode.max_x = inner->bounds.max.x;
    compactNode.max_y = inner->bounds.max.y;
    compactNode.max_z = inner->bounds.max.z;
    compactNode.geomID = CompactBVH2Node::InvalidGeomID;
    compactNode.primID = 0;
    flattenBVH2Tree(inner->children[0], compactTree, depth + 1, index, maxDepth);
    compactNode.secondChildIndex = flattenBVH2Tree(inner->children[1], compactTree, depth + 1, index, maxDepth);
  } else {
    throw std::runtime_error("Unknown derived node encountered during traversal.");
  }

  return myIndex;
}
