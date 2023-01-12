// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains the TriangleMesh class which allows us to represent triangle geometry consistently between IPU and CPU.

#pragma once

#include "CompactBVH2Node.hpp"
#include "Primitives.hpp"
#include "Arrays.hpp"

#ifndef __IPU__
#include <vector>
#endif

struct MeshInfo {
  std::uint32_t firstIndex; // The offset in the index array where this mesh's data starts.
  std::uint32_t firstVertex; // The offset in the vertex array that the offsets refer to.
  std::uint32_t numTriangles; // Number of triangles in this mesh.
  std::uint32_t numVertices; // Number of vertices in this mesh.
};

// This triangle mesh class is templated on the storage type so that we can use dynamic
// storage type (std::vector) on CPU and then easily convert to a static storage
// type (ConstArrayRef) in IPU kernels whilst re-using the same implementation. 
template <template<class T> class Storage>
struct TriangleMesh : Primitive {

  TriangleMesh() {}

  /// Contruct from reference arrays and set bounds.
  TriangleMesh(
      const embree_utils::Bounds3d& _bounds,
      Storage<Triangle>&& externTriangles,
      Storage<embree_utils::Vec3fa>&& externVertices)
    : bounds(_bounds),
      triangles(externTriangles),
      vertices(externVertices)
  {}

  embree_utils::Vec3fa normal(const Intersection& intersection, const embree_utils::Vec3fa&) const override {
    // Record the normal for the current nearest triangle:
    const auto& tri = triangles[intersection.primID];
    const auto& p0 = vertices[tri.v0];
    const auto& p1 = vertices[tri.v1];
    const auto& p2 = vertices[tri.v2];
    return (p1 - p0).cross(p2 - p0).normalized();
  }

  embree_utils::Bounds3d getTriangleBoundingBox(std::uint32_t primID) const {
    const auto& tri = triangles[primID];
    const auto& p0 = vertices[tri.v0];
    const auto& p1 = vertices[tri.v1];
    const auto& p2 = vertices[tri.v2];
    embree_utils::Bounds3d bounds;
    bounds += p0;
    bounds += p1;
    bounds += p2;
    return bounds;
  }

  /// Intersection test with every triangle in the mesh recording nearest (very slow for large meshes):
  Intersection intersect(const embree_utils::Ray& ray) const {
    const RayShearParams transform(ray);
    Intersection result(std::numeric_limits<float>::infinity(), nullptr);

    for (auto primID = 0u; primID < triangles.size(); ++primID) {
      const float t = intersectTriangle(primID, transform, result.t);
      if (t > 0.f && t < result.t) {
        result.primID = primID;
        result.t = t;
        result.prim = this;
      }
    }

    return result;
  }

  /// Intersection test with the triangle specified by primID only:
  Intersection intersect(std::uint32_t primID, const embree_utils::Ray& ray) const override {
    const RayShearParams transform(ray);
    Intersection result(std::numeric_limits<float>::infinity(), nullptr);

    const float t = intersectTriangle(primID, transform, result.t);
    if (t > 0.f && t < result.t) {
      result.primID = primID;
      result.t = t;
      result.prim = this;
    }

    return result;
  }

  embree_utils::Bounds3d getBoundingBox() const override {
    return bounds;
  }

  void updateBoundingBox() {
    bounds = embree_utils::Bounds3d();
    for (const auto& v : vertices) {
      bounds += v;
    }
  }

  float intersectTriangle(std::uint32_t index, const RayShearParams& transform, const float tFar) const;

  embree_utils::Bounds3d bounds;
  Storage<Triangle> triangles;
  Storage<embree_utils::Vec3fa> vertices;
};

#ifndef __IPU__
using HostTriangleMesh = TriangleMesh<std::vector>;
#endif

using CompiledTriangleMesh = TriangleMesh<ConstArrayRef>;
