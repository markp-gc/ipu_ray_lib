// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains the core IPU ray-tracing object that builds and executes
// a ray-tracing graph program for IPU. It conforms to the interfaces in ipu_utils.

#pragma once

#include "ipu_utils.hpp"
#include "Scene.hpp"
#include "Primitives.hpp"

#include <functional>

// IPU ray-tracing graph program builder. The program is designed to
// be comaptible with an Embree/CPU based ray/path tracer for easy
// reference and comparison.
class IpuScene : public ipu_utils::BuilderInterface {
public:
  using RayCallbackFn = std::function<void(std::size_t, const std::vector<embree_utils::TraceResult>&)>;

  IpuScene(const std::vector<Sphere>& _spheres,
           const std::vector<Disc>& _discs,
           SceneRef& sceneRef,
           std::vector<embree_utils::TraceResult>& results,
           std::size_t raysPerWorker,
           RayCallbackFn* fn = nullptr)
    : spheres(_spheres),
      discs(_discs),
      data(sceneRef),
      rayStream(results),
      seedTensor("hw_rng_seed"),
      loopLimit("loop_limit"),
      samplesPerPixel("samples_per_pixel"),
      geometryVar("geom_data"),
      spheresVar("sphere_data"),
      discsVar("disc_data"),
      meshInfoVar("mesh_info"),
      indexBufferVar("index_buffer"),
      vertexBufferVar("vertex_buffer"),
      normalBufferVar("normal_buffer"),
      matIDsVar("matIDs"),
      materialsVar("materials"),
      bvhNodesVar("bvhNodesVar"),
      rayFunc(fn), // If a callback is provided partial results will be streamed to the host.
      numComputeTiles(0u), // This is set in build().
      maxRaysPerWorker(raysPerWorker),
      totalRayBufferSize(0u) // Needs to be set before device to host streams execute.
  {}

  virtual ~IpuScene() {}

  std::size_t getRayStreamSize() const;

  std::vector<std::vector<embree_utils::TraceResult>>& getRayBatches() { return rayBatches; }

  void build(poplar::Graph& graph, const poplar::Target& target) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  double getTraceTimeSecs() const { return traceTimeSecs; }
  RayCallbackFn* getRayCallback() { return rayFunc; }

private:
  const std::vector<Sphere>& spheres;
  const std::vector<Disc>& discs;
  SceneRef data;
  std::vector<embree_utils::TraceResult>& rayStream;

  std::vector<std::uint64_t> seedValues;
  ipu_utils::StreamableTensor seedTensor;
  ipu_utils::StreamableTensor loopLimit;
  ipu_utils::StreamableTensor samplesPerPixel;

  // Variables to hold the primitive data:
  ipu_utils::StreamableTensor geometryVar;
  ipu_utils::StreamableTensor spheresVar;
  ipu_utils::StreamableTensor discsVar;
  ipu_utils::StreamableTensor meshInfoVar;
  ipu_utils::StreamableTensor indexBufferVar;
  ipu_utils::StreamableTensor vertexBufferVar;
  ipu_utils::StreamableTensor normalBufferVar;

  // SceneRef data:
  ipu_utils::StreamableTensor matIDsVar;
  ipu_utils::StreamableTensor materialsVar;
  ipu_utils::StreamableTensor bvhNodesVar;

  poplar::RemoteBuffer rayBuffer;

  std::map<std::string, poplar::Tensor> ioSceneVars;
  std::map<std::string, poplar::Tensor> broadcastSceneVars;
  std::map<std::string, poplar::Tensor> rayTraceVars;

  std::vector<std::vector<embree_utils::TraceResult>> rayBatches;
  RayCallbackFn* rayFunc;

  // This is chosen so that the compute tiles can all
  // be serviced by 32 I/O tiles:
  std::uint32_t dramRayBatches;
  double traceTimeSecs;
  std::size_t numComputeTiles;
  std::size_t maxRaysPerWorker;
  std::size_t totalRayBufferSize;

  poplar::program::Sequence fpSetupProg(poplar::Graph& graph) const;

  std::size_t calcNumBatches(const poplar::Target& target, std::size_t numComputeTiles) const;

  std::vector<std::vector<embree_utils::TraceResult>>
  createRayBatches(const poplar::Device& device, std::size_t numComputeTiles) const;

  void createComputeVars(poplar::Graph& ioGraph,
                         poplar::Graph& computeGraph,
                         std::size_t numComputeTiles,
                         std::size_t sramRayBufferSize);
};
