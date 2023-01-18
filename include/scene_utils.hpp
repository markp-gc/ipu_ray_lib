// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains functionality for loading and building scenes from meshes and other primitives.

#pragma once

#include <vector>
#include <memory>

#include "Primitives.hpp"
#include "Mesh.hpp"
#include "Material.hpp"
#include "xoshiro.hpp"

struct PathTraceSettings {
  xoshiro::Generator sampler;
  std::uint32_t samplesPerPixel;
  std::uint32_t maxPathLength;
  std::uint32_t roulletteStartDepth;
};

struct Camera {
  float horizontalFov;
  // 16 coefficients of homogenous transform matrix in row major order:
  std::vector<float> matrix;
};

struct SceneDescription {
  // Primitives:
  std::vector<HostTriangleMesh> meshes;
  std::vector<Sphere> spheres;
  std::vector<Disc> discs;
  std::vector<Material> materials;

  // Material assignment:
  // ids index the materials above, correspondence with primitives
  // is defined by the order of primitives above.
  std::vector<std::uint32_t> matIDs;

  Camera camera;
  float shadowRayOffset;

  std::unique_ptr<PathTraceSettings> pathTrace;
};

/// Apply a lambda function to every vertex in mesh:
void transform(HostTriangleMesh& mesh, std::function<void(embree_utils::Vec3fa&)>&& tf);

// Functions to build box via code:
void addQuad(HostTriangleMesh& mesh, const std::vector<embree_utils::Vec3fa>& verts);
std::vector<HostTriangleMesh> makeCornellBox();
HostTriangleMesh makeCornellShortBlock();
HostTriangleMesh makeCornellTallBlock();
SceneDescription makeCornellBoxScene(std::string& meshFile, bool boxOnly);

SceneDescription importScene(std::string& filename);
