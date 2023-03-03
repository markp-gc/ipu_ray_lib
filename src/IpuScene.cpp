// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <IpuScene.hpp>
#include <RayCallback.hpp>
#include <serialisation/serialisation.hpp>

#include <poplar/CSRFunctions.hpp>
#include <popops/Loop.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <gcl/TileAllocation.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/Loop.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

#include <xoshiro.hpp>

#include <vector>
#include <random>
#include <algorithm>

IpuScene::IpuScene(const std::vector<Sphere>& _spheres,
          const std::vector<Disc>& _discs,
          SceneRef& sceneRef,
          std::vector<embree_utils::TraceResult>& results,
          std::size_t raysPerWorker,
          RayCallbackFn* fn)
  :
    serialiser(600 * 1024),
    spheres(_spheres),
    discs(_discs),
    data(sceneRef),
    rayStream(results),
    seedTensor("hw_rng_seed"),
    loopLimit("loop_limit"),
    samplesPerPixel("samples_per_pixel"),
    spheresVar("sphere_data"),
    discsVar("disc_data"),
    serialScene("serialScene"),
    rayFunc(fn), // If a callback is provided partial results will be streamed to the host.
    numComputeTiles(0u), // This is set in build().
    maxRaysPerWorker(raysPerWorker),
    totalRayBufferSize(0u) // Needs to be set before device to host streams execute.
{
  // Serialise the scene description for transfer to IPU:
  serialiser << sceneRef.geometry;
  serialiser << sceneRef.meshInfo;
  serialiser << sceneRef.meshTris;
  serialiser << sceneRef.meshVerts;
  serialiser << sceneRef.meshNormals;
  serialiser << sceneRef.matIDs;
  serialiser << sceneRef.materials;
  serialiser << sceneRef.bvhNodes;
  serialiser << sceneRef.maxLeafDepth;
  ipu_utils::logger()->info("Serialised scene size: {} KiB", serialiser.bytes.size() / 1024.f);
}

poplar::program::Sequence IpuScene::fpSetupProg(poplar::Graph& graph) const {
  poplar::program::Sequence prog;
  poplar::FloatingPointBehaviour fpConfig;
  fpConfig.inv = false;
  fpConfig.div0 = false;
  fpConfig.esr = false;
  fpConfig.nanoo = false;
  fpConfig.oflo = false;
  setFloatingPointBehaviour(graph, prog, fpConfig, "setup_fp_behaviour");
  return prog;
}

std::size_t IpuScene::calcNumBatches(const poplar::Target& target, std::size_t numComputeTiles) const {
  // The max size is a multiple of the number of workers by definition:
  const auto numWorkers = target.getNumWorkerContexts();
  const auto maxRaysPerIteration = maxRaysPerWorker * numWorkers * numComputeTiles;
  const auto numReplicas = getRuntimeConfig().numReplicas;
  std::uint32_t numBatches = rayStream.size() / maxRaysPerIteration;

  // If batches does not divide max rays per itr then we need an
  // extra batch to clean up the remainder:
  if (rayStream.size() % maxRaysPerIteration) {
    numBatches += 1;
  }

  // If the number of batches doesn't divide the number of replicas we need
  // extra dummy batches so that all replicas process the same amount of data:
  auto batchCounter = numBatches + (numBatches % numReplicas) * (numReplicas - 1);

  std::uint32_t numBatchesPerReplica = batchCounter / numReplicas;

  ipu_utils::logger()->debug("Ray batches: {}", numBatches);
  ipu_utils::logger()->debug("Ray batches per replica: {}", numBatchesPerReplica);
  ipu_utils::logger()->debug("Loop iterations {} (pipeline loop iterations {})", numBatchesPerReplica, numBatchesPerReplica - 2u);

  // For overlapped I/O we need a pipeline with 2 stages:
  if (numBatchesPerReplica < 2) {
    throw std::runtime_error("Using for async I/O pipeline: number of batches per replica must be at least 2 "
                              "to fill the pipeline.");
  }

  return batchCounter;
}

std::vector<std::vector<embree_utils::TraceResult>>
IpuScene::createRayBatches(const poplar::Device& device, std::size_t numComputeTiles) const {
  // The max size is a multiple of the number of workers by definition:
  const auto numWorkers = device.getTarget().getNumWorkerContexts();
  const auto maxRaysPerIteration = maxRaysPerWorker * numWorkers * numComputeTiles;
  const auto numReplicas = getRuntimeConfig().numReplicas;
  std::uint32_t numBatches = rayStream.size() / maxRaysPerIteration;
  if (rayStream.size() % maxRaysPerIteration) {
    numBatches += 1;
  }

  // Batch the rays using unecessary copying for now:
  std::vector<std::vector<embree_utils::TraceResult>> rayBatches;
  rayBatches.reserve(numBatches);
  auto hitItr = rayStream.begin();

  for (auto i = 0u; i < numBatches; ++i) {
    rayBatches.push_back(std::vector<embree_utils::TraceResult>());
    auto& batch = rayBatches.back();
    batch.reserve(maxRaysPerIteration);
    for (auto j = 0u; j < maxRaysPerIteration; ++j) {
      batch.push_back(*hitItr);
      hitItr += 1;
      if (hitItr == rayStream.end()) {
        break;
      }
    }
  }

  // Pad the last batch:
  if (rayBatches.back().size() != maxRaysPerIteration) {
    const auto padding = maxRaysPerIteration - rayBatches.back().size();
    ipu_utils::logger()->debug("Last batch needs padding: {}", padding);
    // Need to pad with dud rays:
    for (auto i = 0u; i < padding; ++i) {
      // Choose rays that will probably fail at first intersection:
      rayBatches.back().push_back(
        embree_utils::TraceResult(
          embree_utils::HitRecord(embree_utils::Vec3fa(10000, 0, 0), embree_utils::Vec3fa(1, 0, 0)),
          embree_utils::PixelCoord())
      );
    }
    ipu_utils::logger()->debug("Last chunk padded to size: {}", rayBatches.back().size());
  }

  // We also need to pad the number of batches so that all replicas process the same
  // number of batches (same number of loop iterations in the middle of the pipeline):
  const auto remainder = rayBatches.size() % numReplicas;
  if (remainder) {
    auto dummyBatchesNeeded = remainder * (numReplicas - 1);
    ipu_utils::logger()->debug("Padding with {} dummy batches to feed all replicas", dummyBatchesNeeded);
    for (auto d = 0u; d < dummyBatchesNeeded; ++d) {
      rayBatches.push_back(rayBatches.back()); // Just repeat the last batch
    }
  }

  if (rayBatches.size() < 2) {
    throw std::runtime_error("Using for async I/O pipeline: number of batches per replica must be at least 2 "
                              "to fill the pipeline.");
  }

  return rayBatches;
}

/// Create variables in the I/O graph to receive scene data. During initialisation
/// The first tile in the I/O graph receives the data then broadcasts it to all
/// the tiles in the copute graph.
void IpuScene::createComputeVars(poplar::Graph& ioGraph,
                                 poplar::Graph& computeGraph,
                                 std::size_t numComputeTiles,
                                 std::size_t perTileRayBufferSize) {
  // Add a variable to hold each primitive type:
  spheresVar.buildTensor(ioGraph, poplar::UNSIGNED_CHAR, {spheres.size() * sizeof(Sphere)});
  discsVar.buildTensor(ioGraph, poplar::UNSIGNED_CHAR, {discs.size() * sizeof(Disc)});

  // All scene data is serialised into a single byte stream:
  serialScene.buildTensor(ioGraph, poplar::UNSIGNED_CHAR, {serialiser.bytes.size()});

  samplesPerPixel.buildTensor(ioGraph, poplar::UNSIGNED_INT, {1u});

  ipu_utils::logger()->debug("Geometry info: {} bytes per tile", data.geometry.size() * sizeof(GeomRef));
  ipu_utils::logger()->debug("Mesh info: {} bytes per tile", data.meshInfo.size() * sizeof(MeshInfo));
  ipu_utils::logger()->debug("BVH nodes: {} bytes per tile", data.bvhNodes.size() * sizeof(CompactBVH2Node));
  ipu_utils::logger()->debug("Index buffer: {} bytes per tile", data.meshTris.size() * sizeof(Triangle));
  ipu_utils::logger()->debug("Vertex buffer: {} bytes per tile", data.meshVerts.size() * sizeof(embree_utils::Vec3fa));
  ipu_utils::logger()->debug("Normal buffer: {} bytes per tile", data.meshNormals.size() * sizeof(embree_utils::Vec3fa));

  // The scene data vars get uploaded once by the host and then broadcast to every
  // tile so we store them in a separate map.
  // TODO: this should be one monolithic data structure encoded in a single byte tensor.
  // (This will allow us to naturally extend the renderer to allow chunks of BVH and associated
  // data to be loaded piece by piece and for rays to be sorted and sent to tiles that already
  // contain the BVH chunk the rays are traversing next). For now having the separate tensors
  // makes debugging a bit easier as we can see the individual tensors in the profiler.
  ioSceneVars = {
    {"spheres", spheresVar.get()},
    {"discs", discsVar.get()},
    {"meshes", computeGraph.addVariable(poplar::UNSIGNED_CHAR,
                                        {data.meshInfo.size() * sizeof(CompiledTriangleMesh)},
                                        "tri_meshes")},
    {"serialisedScene", serialScene.get()},
    {"samplesPerPixel", samplesPerPixel.get()}
  };

  // Mapping the scene data linearly across I/O tiles
  // is counter productive because it results in excessive
  // exchange messages on the compute tiles so we map each
  // var to its own tile (this won't scale):
  auto tile = 0u;
  for (auto& p : ioSceneVars) {
    ioGraph.setTileMapping(p.second, tile);
    tile += 1;
    tile = tile % ioGraph.getTarget().getNumTiles();
  }

  // These tensors hold the scene data after it has been broadcast to every tile in the compute graph:
  for (const auto& p : ioSceneVars) {
    broadcastSceneVars.insert(std::make_pair(
      p.first,
      computeGraph.addVariable(p.second.elementType(), {numComputeTiles, p.second.numElements()}, p.first + "_broadcast"))
    );
  }

  // Ray data is distributed across tiles:
  rayTraceVars = {
    {"rays", computeGraph.addVariable(poplar::UNSIGNED_CHAR, {numComputeTiles, perTileRayBufferSize}, "sram_ray_buffer")},
  };
}

void IpuScene::build(poplar::Graph& graph, const poplar::Target& target) {
  // Get two disjoint graphs: one set of tiles for compute and
  // another set of tiles for DRAM I/O:
  const auto numTotalTiles = target.getNumTiles() / graph.getReplicationFactor();
  const auto numTilesForIO = gcl::getMinIoTiles(graph);
  ipu_utils::logger()->debug("Reserving {} tiles for I/O", numTilesForIO);
  auto ioTiles = gcl::perIPUTiles(graph, 0, numTilesForIO);
  auto computeTiles = gcl::perIPUTiles(graph, numTilesForIO, numTotalTiles - numTilesForIO);
  auto computeGraph = graph.createVirtualGraph(computeTiles);
  auto ioGraph = graph.createVirtualGraph(ioTiles);

  // Ensure max ray count is a multiple of the number of workers and tiles by definition:
  numComputeTiles = computeGraph.getTarget().getNumTiles();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto maxRaysPerIteration = maxRaysPerWorker * numWorkers;
  const auto perTileRayBufferSize = maxRaysPerIteration * sizeof(embree_utils::TraceResult);
  totalRayBufferSize = numComputeTiles * perTileRayBufferSize;
  ipu_utils::logger()->debug("Num compute tiles: {}", numComputeTiles);
  ipu_utils::logger()->debug("Ray buffer total size: {}", totalRayBufferSize);
  ipu_utils::logger()->debug("Ray buffer: {} bytes per tile", perTileRayBufferSize);

  // Optimise stream copies to reduce memory use:
  const bool optimiseMemUse = true;

  // Add the required codelets:
  computeGraph.addCodelets("TraceCodelets.gp");
  popops::addCodelets(computeGraph);
  poprand::addCodelets(computeGraph);

  createComputeVars(ioGraph, computeGraph, numComputeTiles, perTileRayBufferSize);

  // Add remote buffer to store ray-batches:
  auto rayBatchesPerReplica = calcNumBatches(target, numComputeTiles) / getRuntimeConfig().numReplicas;
  ipu_utils::logger()->debug("Allocating buffer for {} ray batches in DRAM.", rayBatchesPerReplica);
  ipu_utils::logger()->debug("DRAM Ray-buffer size: {} bytes", totalRayBufferSize * rayBatchesPerReplica);
  rayBuffer = computeGraph.addRemoteBuffer("dram_ray_buffer", poplar::UNSIGNED_CHAR,
                                    totalRayBufferSize, rayBatchesPerReplica,
                                    true, optimiseMemUse);

  // Add a duplicate SRAM ray buffer distributed across all I/O tiles:
  auto loadBuffer = ioGraph.addVariable(poplar::UNSIGNED_CHAR, {totalRayBufferSize},
                                        poplar::VariableMappingMethod::LINEAR, "ray_load_buffer");
  // We also need another tmp duplicate on the IO tiles:
  auto saveBuffer = ioGraph.addVariable(poplar::UNSIGNED_CHAR, {totalRayBufferSize},
                                        poplar::VariableMappingMethod::LINEAR, "ray_save_buffer");
  // Loop counter variables belong in I/O graph:
  auto loadIndex = ioGraph.addVariable(poplar::UNSIGNED_INT, {}, "load_index");
  auto saveIndex = ioGraph.addVariable(poplar::UNSIGNED_INT, {}, "save_index");
  loopLimit.buildTensor(ioGraph, poplar::UNSIGNED_INT, {});
  ioGraph.setTileMapping(loadIndex, 0);
  ioGraph.setTileMapping(saveIndex, 0);
  ioGraph.setTileMapping(loopLimit, 0);

  // Reads/writes from DRAM to I/O tiles and
  // reads/writes from IO tiles to compute tiles:
  auto loadRaysFromDRAM = poplar::program::Copy(rayBuffer, loadBuffer, loadIndex, "load_rays");
  auto copyRaysIOToCompute = poplar::program::Copy(loadBuffer, rayTraceVars["rays"].flatten());
  auto copyRaysComputeToIO = poplar::program::Copy(rayTraceVars["rays"].flatten(), saveBuffer);
  // Stream the save index back to the host so it can monitor progress:
  auto hostRayStream = ioGraph.addDeviceToHostFIFO("save_rays", saveBuffer.elementType(), saveBuffer.numElements());
  poplar::program::Sequence saveRays = {
    // Save rays to DRAM:
    poplar::program::Copy(saveBuffer, rayBuffer, saveIndex, "save_rays"),
    // Also stream saved rays to host:
    poplar::program::Copy(saveBuffer, hostRayStream, "rays_to_host"),
  };

  // Compute sets:
  auto initCs = computeGraph.addComputeSet("init_data_cs");
  auto traceCs = computeGraph.addComputeSet("trace_cs");

  for (auto t = 0u; t < numComputeTiles; ++t) {
    // Some host side objects contain pointers (e.g. vtables). Hence, they are not
    // compatible with IPU so they need to re-newed (with placement new) on the IPU.
    // Note: plain old data (POD) structs can usually be made comaptible with IPU so
    // some other data arrays are not reinitialised.
    auto initBuildVertex = computeGraph.addVertex(initCs, "BuildDataStructures");
    computeGraph.setTileMapping(initBuildVertex, t);
    computeGraph.connect(initBuildVertex["spheres"], broadcastSceneVars["spheres"][t]);
    computeGraph.connect(initBuildVertex["discs"], broadcastSceneVars["discs"][t]);
    computeGraph.connect(initBuildVertex["meshes"], broadcastSceneVars["meshes"][t]);

    // Scene buffer:
    computeGraph.connect(initBuildVertex["serialisedScene"], broadcastSceneVars["serialisedScene"][t]);

    // Ray tracing:
    // Choose between two ray trace modes at compile time.
    poplar::VertexRef rayTraceVertex;
    if (data.pathTrace) {
      rayTraceVertex = computeGraph.addVertex(traceCs, "PathTrace");
      computeGraph.setInitialValue(rayTraceVertex["maxPathLength"], data.maxPathLength);
      computeGraph.setInitialValue(rayTraceVertex["rouletteStartDepth"], data.rouletteStartDepth);
      computeGraph.setInitialValue(rayTraceVertex["imageWidth"], data.imageWidth);
      computeGraph.setInitialValue(rayTraceVertex["imageHeight"], data.imageHeight);
      computeGraph.setInitialValue(rayTraceVertex["antiAliasScale"], data.antiAliasScale);
      computeGraph.setInitialValue(rayTraceVertex["fovRadians"], data.fovRadians);
      computeGraph.connect(rayTraceVertex["samplesPerPixel"], broadcastSceneVars["samplesPerPixel"][t][0]);
    } else {
      rayTraceVertex = computeGraph.addVertex(traceCs, "ShadowTrace");
      computeGraph.setInitialValue(rayTraceVertex["ambientLightFactor"], .05f);
      embree_utils::Vec3fa lightPos(18, 257, -1060);
      auto lp = computeGraph.addConstant(poplar::FLOAT, {3}, &lightPos.x);
      computeGraph.setTileMapping(lp, t);
      computeGraph.connect(rayTraceVertex["lightPos"], lp);
    }

    computeGraph.connect(rayTraceVertex["rays"], rayTraceVars["rays"][t]);

    computeGraph.setTileMapping(rayTraceVertex, t);
    computeGraph.connect(rayTraceVertex["spheres"], broadcastSceneVars["spheres"][t]);
    computeGraph.connect(rayTraceVertex["discs"], broadcastSceneVars["discs"][t]);
    computeGraph.connect(rayTraceVertex["meshes"], broadcastSceneVars["meshes"][t]);
    computeGraph.connect(rayTraceVertex["serialisedScene"], broadcastSceneVars["serialisedScene"][t]);
    computeGraph.setInitialValue(rayTraceVertex["maxLeafDepth"], data.maxLeafDepth);

    // Set tile mappings:
    for (auto& p : broadcastSceneVars) {
      computeGraph.setTileMapping(p.second[t], t);
    }

    for (auto& p : rayTraceVars) {
      computeGraph.setTileMapping(p.second[t], t);
    }
  }

  // Add a program to broadcast the scene data to all compute tiles:
  std::map<std::string, poplar::Tensor> varReceivingSlices;

  ipu_utils::logger()->info("Building scene broadcast");
  poplar::program::Sequence broadcastSceneData;
  for (auto& p : ioSceneVars) {
    auto src = p.second; // Data is received to this slice
    auto dst = broadcastSceneVars.at(p.first);
    auto numReceiving = dst.dim(0);
    auto srcBroadcast = src.reshape({1, src.numElements()}).broadcast(numReceiving, 0); // Create a broadcast view
    broadcastSceneData.add(poplar::program::Copy(srcBroadcast, dst)); // Copy the broadcast view to the rest of the tiles
  }

  // Free up the space used to stream the scene data from the host:
  for (auto& p : ioSceneVars) {
    auto src = p.second; // Data is received to this slice
    broadcastSceneData.add(poplar::program::WriteUndef(p.second, "undef_" + p.first)); // Copy the broadcast view to the rest of the tiles
  }

  ipu_utils::logger()->info("Building init sequence");
  poplar::program::Sequence init = {
    fpSetupProg(computeGraph),
    spheresVar.buildWrite(computeGraph, optimiseMemUse),
    discsVar.buildWrite(computeGraph, optimiseMemUse),
    serialScene.buildWrite(ioGraph, optimiseMemUse),
    samplesPerPixel.buildWrite(computeGraph, optimiseMemUse),
    broadcastSceneData
  };

  // Add program to set HW RNG seed to the init sequence:
  seedTensor.buildTensor(computeGraph, poplar::UNSIGNED_INT, {2});
  computeGraph.setTileMapping(seedTensor, 0);
  // Do not broadcast the seed to all replicas: need a different seed per replica:
  init.add(seedTensor.buildWrite(computeGraph, optimiseMemUse, false));
  poprand::setSeed(computeGraph, seedTensor, 1u, init, "set_seed");

  // Final part of initialisation is to run the init compute set:
  init.add(poplar::program::Execute({initCs}));

  // Program to initialise the I/O pipeline loop counter:
  poplar::program::Sequence initLoopCounters {
    loopLimit.buildWrite(ioGraph, optimiseMemUse)
  };

  // Adjust loop limit for async I/O pipeline:
  popops::subInPlace(ioGraph, loopLimit, 2u, initLoopCounters, "loop_limit_init");
  init.add(initLoopCounters);

  // Make our own loop (instead of e.g. using countedForLoop) so we can ensure asynchronous I/O:
  ioGraph.setInitialValue(saveIndex, 0u);
  ioGraph.setInitialValue(loadIndex, 0u);
  poplar::program::Sequence incLoadIndex;
  popops::addInPlace(ioGraph, loadIndex, 1u, incLoadIndex, "increment_load_index");
  poplar::program::Sequence incSaveIndex;
  popops::addInPlace(ioGraph, saveIndex, 1u, incSaveIndex, "increment_save_index");

  poplar::program::Sequence cond;
  poplar::Tensor pred = popops::sub(ioGraph, loopLimit.get(), saveIndex, cond, "calc_loop_condition");
  cond.add(poplar::program::AssumeEqualAcrossReplicas(pred, "pred_assume_equal"));

  poplar::program::Sequence rayTraceBody {
    poplar::program::Execute(traceCs)
  };

  // Main loop should read and write DRAM
  // asynchronously with compute:
  poplar::program::Sequence traceBody {
    //poplar::program::PrintTensor("save idx: ", saveIndex),
    saveRays,
    incSaveIndex,
    incLoadIndex,
    //poplar::program::PrintTensor("load idx: ", loadIndex),
    loadRaysFromDRAM,
    rayTraceBody,
    copyRaysComputeToIO,
    copyRaysIOToCompute
  };

  // This is the main loop over ray-batches:
  auto traceLoop = poplar::program::RepeatWhileTrue(cond, pred, traceBody, "trace_loop");

  poplar::program::Sequence pipeline {
    //poplar::program::PrintTensor("load idx: ", loadIndex),
    loadRaysFromDRAM,
    copyRaysIOToCompute,
    rayTraceBody,
    incLoadIndex,
    //poplar::program::PrintTensor("load idx: ", loadIndex),
    loadRaysFromDRAM,
    copyRaysComputeToIO,
    copyRaysIOToCompute,
    traceLoop,
    //poplar::program::PrintTensor("save idx: ", saveIndex),
    saveRays,
    incSaveIndex,
    rayTraceBody,
    copyRaysComputeToIO,
    //poplar::program::PrintTensor("save idx: ", saveIndex),
    saveRays,
  };

  poplar::program::Sequence trace = {
    init,
    pipeline
  };

  getPrograms().add("trace", trace);
}

void IpuScene::execute(poplar::Engine& engine, const poplar::Device& device) {
  // If ray list is larger than max size per iteration then we need to
  // serialise it into smaller chunks. We also need to make sure each chunk is
  // a multiple of the number of workers.

  // The max size is a multiple of the number of workers by definition:
  const auto numWorkers = device.getTarget().getNumWorkerContexts();
  const auto totalRaysPerIteration = numComputeTiles * maxRaysPerWorker * numWorkers;
  totalRayBufferSize = totalRaysPerIteration * sizeof(embree_utils::TraceResult);
  const auto numReplicas = getRuntimeConfig().numReplicas;
  ipu_utils::logger()->debug("Rays per iteration: {} Total rays: {} ", totalRaysPerIteration, rayStream.size());

  rayBatches = createRayBatches(device, numComputeTiles);
  std::uint32_t numBatchesPerReplica = rayBatches.size() / numReplicas;
  loopLimit.connectWriteStream(engine, &numBatchesPerReplica);
  samplesPerPixel.connectWriteStream(engine, &data.samplesPerPixel);

  // Set a different RNG seed per-replica:
  xoshiro::State s;
  xoshiro::seed(s, data.rngSeed);
  for (auto r = 0u; r < numReplicas; ++r) {
    seedValues.push_back(xoshiro::next128ss(s));
  }
  seedTensor.connectWriteStream(engine, seedValues);

  // Note: these copy host pointers into the on device data! We will
  // overwrite by rebuilding the object before using it in the codelet:
  spheresVar.connectWriteStream(engine, (void*)spheres.data());
  discsVar.connectWriteStream(engine, (void*)discs.data());
  serialScene.connectWriteStream(engine, (void*)serialiser.bytes.data());

  // Connect callbacks:
  for (auto i = 0u; i < numReplicas; ++i) {
    engine.connectStreamToCallback("save_rays", i, std::make_unique<RayCallback>(*this, i));
  }

  // We don't include sending initial rays to DRAM in timing (they
  // would stay there for the duration of a real render)
  auto startTime = std::chrono::steady_clock::now();
  std::size_t replicaIndices[numReplicas] = {0};
  for (auto i = 0u; i < rayBatches.size(); ++i) {
    auto& v = rayBatches[i];
    if (v.size() != totalRaysPerIteration) {
      throw std::runtime_error("The number of rays in each batch is not correct: " + std::to_string(v.size()));
    }
    const auto replica = i % numReplicas; // Cycle through the replicas for each sequential batch index
    engine.copyToRemoteBuffer(v.data(), "dram_ray_buffer", replicaIndices[replica], replica);
    replicaIndices[replica] += 1;
  }
  auto endTime = std::chrono::steady_clock::now();
  auto secs = std::chrono::duration<double>(endTime - startTime).count();
  ipu_utils::logger()->info("Host to DRAM ray bandwidth: {} GB/sec", (1e-9 * rayBatches.size() * totalRaysPerIteration * sizeof(embree_utils::TraceResult) / secs));

  // Include initialisation (send BVH to device) in IPU timings:
  ipu_utils::logger()->info("IPU Rendering started.");
  startTime = std::chrono::steady_clock::now();
  getPrograms().run(engine, "trace");
  endTime = std::chrono::steady_clock::now();
  traceTimeSecs = std::chrono::duration<double>(endTime - startTime).count();
  ipu_utils::logger()->info("IPU Rendering finished.");

  if (rayFunc == nullptr) {
    // No callback was set so read rays back to host from DRAM in bulk:
    startTime = std::chrono::steady_clock::now();
    for (auto& i : replicaIndices) { i = 0; }
    for (auto i = 0u; i < rayBatches.size(); i += 1) {
      const auto replica = i % numReplicas;
      engine.copyFromRemoteBuffer("dram_ray_buffer", rayBatches[i].data(), replicaIndices[replica], replica);
      replicaIndices[replica] += 1;
    }
    endTime = std::chrono::steady_clock::now();
    secs = std::chrono::duration<double>(endTime - startTime).count();
    ipu_utils::logger()->info("DRAM to host ray bandwidth: {} GB/sec", (1e-9 * rayBatches.size() * totalRaysPerIteration * sizeof(embree_utils::TraceResult) / secs));
  }

  // Copy result back into original stream:
  auto hitItr = rayStream.begin();
  auto index = 0u;
  for (const auto& v : rayBatches) {
    auto count = 0u;
    for (auto& h : v) {
      *hitItr = h;
      hitItr += 1;
      count += 1;
      if (hitItr == rayStream.end()) {
        break;
      }
    }
    index += 1;
    if (hitItr == rayStream.end()) {
      break;
    }
  }
}

/// Return the per batch ray stream size for use when copying the ray stream
/// to the host. The value gets set in build() and execute() because it requires
/// knowledge of the target.
std::size_t IpuScene::getRayStreamSize() const {
  return totalRayBufferSize;
}
