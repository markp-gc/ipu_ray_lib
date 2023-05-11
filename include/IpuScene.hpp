// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains the core IPU ray-tracing object that builds and executes
// a ray-tracing graph program for IPU. It conforms to the interfaces in ipu_utils.

#pragma once

#include "ipu_utils.hpp"
#include "Scene.hpp"
#include "Primitives.hpp"
#include <serialisation/Serialiser.hpp>

#include <poplin/MatMul.hpp>

#include <functional>

class NifModel;

// IPU ray-tracing graph program builder. The program is designed to
// be comaptible with an Embree/CPU based ray/path tracer for easy
// reference and comparison.
class IpuScene : public ipu_utils::BuilderInterface {

  struct NifResult {
    poplar::Tensor bgr;
    poplar::program::Sequence init;
    poplar::program::Sequence exec;
  };

public:
  using RayCallbackFn = std::function<void(std::size_t, const std::vector<embree_utils::TraceResult>&)>;

  IpuScene(const std::vector<Sphere>& _spheres,
           const std::vector<Disc>& _discs,
           SceneRef& sceneRef,
           std::vector<embree_utils::TraceResult>& results,
           std::size_t raysPerWorker,
           RayCallbackFn* fn = nullptr);

  virtual ~IpuScene();

  std::size_t getRayStreamSize() const;

  std::vector<std::vector<embree_utils::TraceResult>>& getRayBatches() { return rayBatches; }

  bool loadNifModel(const std::string& assetPath);
  NifResult buildNifHdri(poplar::Graph& g, std::unique_ptr<NifModel>& nif, poplar::Tensor uvs);
  void setHdriRotation(float degrees);
  void setAvailableMemoryProportion(float proportion);
  void setMaxNifBatchSize(std::size_t raysPerBatch);

  void build(poplar::Graph& graph, const poplar::Target& target) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  double getTraceTimeSecs() const { return traceTimeSecs; }
  RayCallbackFn* getRayCallback() { return rayFunc; }

private:
  Serialiser<16> serialiser;
  const std::vector<Sphere>& spheres;
  const std::vector<Disc>& discs;
  SceneRef data;
  std::vector<embree_utils::TraceResult>& rayStream;

  std::vector<std::uint64_t> seedValues;
  ipu_utils::StreamableTensor seedTensor;
  ipu_utils::StreamableTensor loopLimit;
  ipu_utils::StreamableTensor samplesPerPixel;
  ipu_utils::StreamableTensor azimuthRotation;

  // Variables to hold the primitive data:
  ipu_utils::StreamableTensor spheresVar;
  ipu_utils::StreamableTensor discsVar;

  // SceneRef data in serialised form:
  ipu_utils::StreamableTensor serialScene;

  poplar::RemoteBuffer rayBuffer;

  std::map<std::string, poplar::Tensor> ioSceneVars;
  std::map<std::string, poplar::Tensor> broadcastSceneVars;
  std::map<std::string, poplar::Tensor> rayTraceVars;

  std::unique_ptr<NifModel> nif;
  poplin::matmul::PlanningCache cache;

  std::vector<std::vector<embree_utils::TraceResult>> rayBatches;
  RayCallbackFn* rayFunc;

  // This is chosen so that the compute tiles can all
  // be serviced by 32 I/O tiles:
  std::uint32_t dramRayBatches;
  double traceTimeSecs;
  std::size_t numComputeTiles;
  std::size_t maxRaysPerWorker;
  std::size_t totalRayBufferSize;
  float hdriRotationDegrees;
  float nifMemoryProportion;
  std::size_t nifMaxRaysPerBatch;

  poplar::program::Sequence fpSetupProg(poplar::Graph& graph) const;

  std::size_t calcNumBatches(const poplar::Target& target, std::size_t numComputeTiles) const;

  std::vector<std::vector<embree_utils::TraceResult>>
  createRayBatches(const poplar::Device& device, std::size_t numComputeTiles) const;

  void createComputeVars(poplar::Graph& ioGraph,
                         poplar::Graph& computeGraph,
                         std::size_t numComputeTiles,
                         std::size_t maxRaysPerIteration);
};
