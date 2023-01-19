// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains IPU compute codelets (kernels) for ray/path-tracing.

#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <poplar/StackSizeDefs.hpp>

#include <Arrays.hpp>
#include <Primitives.hpp>
#include <Mesh.hpp>
#include <Scene.hpp>
#include <CompactBvh.hpp>
#include <Material.hpp>
#include <Render.hpp>
#include <BxDF.hpp>
#include <embree_utils/geometry.hpp>

#include <new>

#include "sincos.hpp"

using namespace poplar;
using namespace embree_utils;

// Manually set the stack size for the codelets. We need to do this
// because we have been lazy by recording the BVH traversal on the
// worker stacks. TODO: connect a tensor so graph construction
// guarantees space for the BVH traversal (we know the max depth
// of the tree at compute graph construction/compile time):
DEF_STACK_USAGE(960, __runCodelet_PathTrace);
DEF_STACK_USAGE(960, __runCodelet_ShadowTrace);

// Utility to get a uniform sample between 0 and 1
// from the IPU's hardware RNG:
inline float hw_uniform_0_1() {
  return __builtin_ipu_urand_f32() + .5f;
}

/// Some objects have been transferred direct from CPU to IPU but
/// will contain incomptible pointer data but compatible plain old data.
/// This vertex re-allocates these objects/structures for the IPU.
/// Note: this is very questionable but it simplifies things considerably.
/// At some point this will vanish anyway as the scene description will be
/// organised into opaque binary chunks so that we can page chunks in and
/// out of SRAM:
class BuildDataStructures : public Vertex {
public:
  // Individual primitive arrays:
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Sphere)>> spheres;
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Disc)>> discs;
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(MeshInfo)>> meshInfo;

  // Mesh internal data arrays:
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Vec3fa)>> verts;
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Triangle)>> tris;
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(CompiledTriangleMesh)>> meshes;

  bool compute() {
    // For each primitive copy its position to a tensor:
    auto wrappedSpheres = ConstArrayRef<Sphere>::reinterpret(&spheres[0], spheres.size());
    auto wrappedDiscs = ConstArrayRef<Disc>::reinterpret(&discs[0], discs.size());
    auto wrappedMeshInfo = ConstArrayRef<MeshInfo>::reinterpret(&meshInfo[0], meshInfo.size());
    auto wrappedVerts = ConstArrayRef<Vec3fa>::reinterpret(&verts[0], verts.size());
    auto wrappedTris = ConstArrayRef<Triangle>::reinterpret(&tris[0], tris.size());

    // Need to re-new everything i.e. reconstruct in place using placement-new!
    // Any struct with pointers (e.g. vtable) will not be compatible between IPU and host.
    // Hopefully one day codelets like this can be auto-generated.
    //
    // There are some assumptions that make this work:
    // 1. Host pointers are larger than IPU pointers so everything has been over allocated.
    // 2. V-tables come at the end so other data has same layout.
    // 3. Tensors are not moved on IPU between this vertex and the trace vertex running
    //    (we are storing pointers for reuse between different vertices!)
    // Basically this is all very dubious but works for now...

    // Mesh objects are constructed from the mesh info data:
    auto wrappedMeshes = ConstArrayRef<CompiledTriangleMesh>::reinterpret(&meshes[0], meshes.size());
    auto meshIdx = 0u;
    for (const auto& info : wrappedMeshInfo) {
      new ((void*)&wrappedMeshes[meshIdx]) CompiledTriangleMesh(
        embree_utils::Bounds3d(),
        ConstArrayRef(&wrappedTris[info.firstIndex], info.numTriangles),
        ConstArrayRef(&wrappedVerts[info.firstVertex], info.numVertices)
      );
      meshIdx += 1;
    }

    // Other primitves are "re-newed" using their own data:
    for (auto &s : wrappedSpheres) {
      new ((void*)&s) Sphere(s.centre, s.radius);
    }

    for (auto &d : wrappedDiscs) {
      new ((void*)&d) Disc(d.n, d.c, d.r);
    }

    return false;
  }
};

// Look up the underlying primitive from a geometry type and ID:
const Primitive* getPrimitive(
  const GeomRef& geom,
  const ConstArrayRef<Sphere>& spheres,
  const ConstArrayRef<CompiledTriangleMesh>& meshes,
  const ConstArrayRef<Disc>& discs
) {
  switch (geom.type) {
    case GeomType::Mesh:
      return &meshes[geom.index];
    case GeomType::Sphere:
      return &spheres[geom.index];
    case GeomType::Disc:
      return &discs[geom.index];
    case GeomType::NumTypes:
    default:
      return nullptr;
  }
}

void sampleCameraRays(int workerID,
                      float imageWidth, float imageHeight,
                      float2 antiAliasScale, float fovRadians,
                      ArrayRef<embree_utils::TraceResult>& wrappedRays) {
  // Do trig outside of loop:
  float s, c;
  sincos(fovRadians / 2.f, s, c);
  const auto fovTanTheta = s / c;
  const auto rayOrigin = embree_utils::Vec3fa(0.f, 0.f, 0.f);

  // Generate camera rays. Each worker starts processing offset by their worker IDs.
  // The external Poplar graph construction code ensures the number of rays to process on each
  // tile is a multiple of 6 (by padding or otherwise):
  for (auto r = workerID; r < wrappedRays.size(); r += poplar::MultiVertex::numWorkers()) {
    auto& result = wrappedRays[r];
    // Sample around the pixel coord in the ray stream (anti-aliasing):
    float2 g = __builtin_ipu_f32v2grand();
    float2 p = {result.p.u, result.p.v}; // row, col
    p += antiAliasScale * g;
    const auto rayDir = pixelToRayDir(p[1], p[0], imageWidth, imageHeight, fovTanTheta);
    result.h = embree_utils::HitRecord(rayOrigin, rayDir);
  }
}

/// Simple uni-directional path trace vertex. Rays are path traced one by one
/// alternating BVH intersection and BxDF sampling to produce the incoming ray
/// direction. There is no light sampling so we rely on hitting light sources by
/// chance.
class PathTrace : public MultiVertex {
public:
  // Storage for sphere, disc, and mesh primitives:
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Sphere)>> spheres;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Disc)>> discs;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(CompiledTriangleMesh)>> meshes;

  // Index and vertex buffers:
  // NOTE: Even though these aren't referenced in the codelet they need to be connected
  // and kept live because the mesh holds pointers to their connected tensors.
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Triangle)>> tris;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Vec3fa)>> verts;

  // Scene description and BVH:
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(GeomRef)>> geometry;
  Input<Vector<unsigned int, poplar::VectorLayout::SPAN>> matIDs;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Material)>> materials;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(CompactBVH2Node)>> bvhNodes;

  // Max depth needed for the BVH traversal stack:
  Input<std::uint32_t> samplesPerPixel;
  std::uint32_t maxLeafDepth;
  std::uint32_t maxPathLength;
  std::uint32_t rouletteStartDepth;
  float imageWidth;
  float imageHeight;
  float antiAliasScale;
  float fovRadians;

  // Ray stream:
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(TraceResult)>> rays;

  bool compute(unsigned int workerID) {
    // Wrap all byte arrays with their correct types:
    auto wrappedSpheres = ConstArrayRef<Sphere>::reinterpret(&spheres[0], spheres.size());
    auto wrappedMeshes = ConstArrayRef<CompiledTriangleMesh>::reinterpret(&meshes[0], meshes.size());
    auto wrappedDiscs = ConstArrayRef<Disc>::reinterpret(&discs[0], discs.size());
    auto wrappedGeometry = ConstArrayRef<GeomRef>::reinterpret(&geometry[0], geometry.size());
    auto wrappedBvhNodes = ConstArrayRef<CompactBVH2Node>::reinterpret(&bvhNodes[0], bvhNodes.size());
    auto wrappedRays = ArrayRef<embree_utils::TraceResult>::reinterpret(&rays[0], rays.size());
    auto wrappedMaterials = ConstArrayRef<Material>::reinterpret(&materials[0], materials.size());
    auto wrappedMatIDs = ConstArrayRef<unsigned int>::reinterpret(&matIDs[0], matIDs.size());

    // Construct a BVH from all the wrapped arrays:
    CompactBvh bvh(wrappedBvhNodes, maxLeafDepth);

    auto primLookup = [&](std::uint16_t geomID, std::uint32_t primID) {
      const auto& geom = wrappedGeometry[geomID];
      return getPrimitive(geom, wrappedSpheres, wrappedMeshes, wrappedDiscs);
    };

    for (auto s = 0u; s < samplesPerPixel; ++s) {
      // Generate ray samples:
      sampleCameraRays(workerID, imageWidth, imageHeight,
                       float2{antiAliasScale, antiAliasScale},
                       fovRadians, wrappedRays);

      // Intersect all rays against the BVH. Each worker starts processing offset by their worker IDs.
      // The external Poplar graph construction code ensures the number of rays to process on each
      // tile is a multiple of 6 (by padding or otherwise):
      for (auto r = workerID; r < wrappedRays.size(); r += numWorkers()) {
        auto& result = wrappedRays[r];
        auto& hit = result.h;
        Vec3fa throughput(1.f, 1.f, 1.f);
        Vec3fa color(0.f, 0.f, 0.f);

        for (auto i = 0u; i < maxPathLength; ++i) {
          offsetRay(hit.r, hit.normal); // offset rays to avoid self intersection.
          // Reset ray limits for next bounce:
          hit.r.tMin = 0.f;
          hit.r.tMax = std::numeric_limits<float>::infinity();
          auto intersected = bvh.intersect(hit.r, primLookup);

          if (intersected) {
            updateHit(intersected, hit);
            const auto& material = wrappedMaterials[wrappedMatIDs[hit.geomID]];

            if (material.emissive) {
              color += throughput * material.emission;
            }

            if (material.type == Material::Type::Diffuse) {
              // Use HW random number generator for samples:
              const float u1 = hw_uniform_0_1();
              const float u2 = hw_uniform_0_1();
              hit.r.direction = sampleDiffuse(hit.normal, u1, u2);
              // Update throughput
              //const float w = std::abs(wiWorld.dot(normal));
              //const float pdf = cosineHemispherePdf(wiTangent);
              // The terms w / (Pi * pdf) all cancel for diffuse throughput:
              throughput *= material.albedo; // * (w / (Pi * pdf)); // PDF terms cancel for cosine weighted samples
              //throughput *= material.albedo * (wiTangent.z * 2.f); // Apply PDF for hemisphere samples (sampleDir is in tangent space so cos(theta) == z-coord).
            } else if (material.type == Material::Type::Specular) {
              hit.r.direction = reflect(hit.r.direction, hit.normal);
              throughput *= material.albedo;
            } else if (material.type == Material::Type::Refractive) {
              const float u1 = hw_uniform_0_1();
              const auto [dir, refracted] = dielectric(hit.r, hit.normal, material.ior, u1);
              hit.r.direction = dir;
              if (refracted) { throughput *= material.albedo; }
            } else {
              // Mark an error:
              result.rgb *= std::numeric_limits<float>::quiet_NaN();
            }
          } else {
            break;
          }

          // Random stopping:
          if (i > rouletteStartDepth) {
            const float u1 = hw_uniform_0_1();
            if (evaluateRoulette(u1, throughput)) { break; }
          }
        }

        result.rgb += color;
      }

    } // end of sample loop

    return true;
  }
};

/// Simple ray trace vertex primarily intended for testing and validation. The vertex intersects
/// every ray with the BVH to get primary hits and then traces one shadow ray from each hit
/// to a fixed point light source.
class ShadowTrace : public MultiVertex {
public:

  // Storage for sphere, disc, and mesh primitives:
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Sphere)>> spheres;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Disc)>> discs;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(CompiledTriangleMesh)>> meshes;

  // Index and vertex buffers:
  // NOTE: Even though these aren't referenced in the codelet they need to be connected
  // and kept live because the mesh holds pointers to their connected tensors.
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Triangle)>> tris;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Vec3fa)>> verts;

  // Scene description and BVH:
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(GeomRef)>> geometry;
  Input<Vector<unsigned int, poplar::VectorLayout::SPAN>> matIDs;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(Material*)>> materials;
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(CompactBVH2Node)>> bvhNodes;

  // Max depth needed for the BVH traversal stack:
  std::uint32_t maxLeafDepth;
  float ambientLightFactor;
  Input<Vector<float>> lightPos;

  // Ray stream:
  InOut<Vector<unsigned char, poplar::VectorLayout::SPAN, alignof(TraceResult)>> rays;

  bool compute(unsigned int workerID) {
    // Wrap all byte arrays with their correct types:
    auto wrappedSpheres = ConstArrayRef<Sphere>::reinterpret(&spheres[0], spheres.size());
    auto wrappedMeshes = ConstArrayRef<CompiledTriangleMesh>::reinterpret(&meshes[0], meshes.size());
    auto wrappedDiscs = ConstArrayRef<Disc>::reinterpret(&discs[0], discs.size());
    auto wrappedGeometry = ConstArrayRef<GeomRef>::reinterpret(&geometry[0], geometry.size());
    auto wrappedBvhNodes = ConstArrayRef<CompactBVH2Node>::reinterpret(&bvhNodes[0], bvhNodes.size());
    auto wrappedRays = ArrayRef<embree_utils::TraceResult>::reinterpret(&rays[0], rays.size());
    auto wrappedMaterials = ConstArrayRef<Material>::reinterpret(&materials[0], materials.size());
    auto wrappedMatIDs = ConstArrayRef<unsigned int>::reinterpret(&matIDs[0], matIDs.size());

    // Construct a BVH from all the wrapped arrays:
    CompactBvh bvh(wrappedBvhNodes, maxLeafDepth);

    auto primLookup = [&](std::uint16_t geomID, std::uint32_t primID) {
      const auto& geom = wrappedGeometry[geomID];
      return getPrimitive(geom, wrappedSpheres, wrappedMeshes, wrappedDiscs);
    };

    Vec3fa lp(lightPos[0], lightPos[1], lightPos[2]);

    // There is no need for ray gen in this vertex since we are tracibg rays
    // that were generated on the host.

    // Intersect all rays against the BVH. Each worker starts processing offset by their worker IDs.
    // The external Poplar graph construction code ensures the number of rays to process on each
    // tile is a multiple of 6 (by padding or otherwise):
    for (auto r = workerID; r < wrappedRays.size(); r += numWorkers()) {
      auto& result = wrappedRays[r];

      traceShadowRay(
        bvh,
        wrappedMatIDs, wrappedMaterials,
        ambientLightFactor,
        result, primLookup, lp);
    }

    return true;
  }
};