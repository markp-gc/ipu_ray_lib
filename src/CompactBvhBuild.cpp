// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <CompactBvhBuild.hpp>

template <typename T>
CompactBVH2Node toCompactNode(const T& node) {
  CompactBVH2Node compactNode;
  compactNode.min_x = node.bounds.min.x;
  compactNode.min_y = node.bounds.min.y;
  compactNode.min_z = node.bounds.min.z;
  compactNode.dx = CompactBVH2Node::OffsetType(node.bounds.max.x - node.bounds.min.x);
  compactNode.dy = CompactBVH2Node::OffsetType(node.bounds.max.y - node.bounds.min.y);
  compactNode.dz = CompactBVH2Node::OffsetType(node.bounds.max.z - node.bounds.min.z);

  if constexpr (std::is_same_v<embree_utils::LeafNode, T>) {
    compactNode.geomID = node.geomID;
    compactNode.primID = node.primID;
  } else {
    compactNode.geomID = CompactBVH2Node::InvalidGeomID;
    compactNode.primID = 0;
  }
  return compactNode;
}

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
    compactNode = toCompactNode(*leaf);
  } else if (auto* inner = dynamic_cast<const embree_utils::InnerNode*>(node)) {
    compactNode = toCompactNode(*inner);
    flattenBVH2Tree(inner->children[0], compactTree, depth + 1, index, maxDepth);
    compactNode.secondChildIndex = flattenBVH2Tree(inner->children[1], compactTree, depth + 1, index, maxDepth);
  } else {
    throw std::runtime_error("Unknown derived node encountered during traversal.");
  }

  return myIndex;
}
