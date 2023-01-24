// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <RayCallback.hpp>
#include <IpuScene.hpp>

RayCallback::RayCallback(IpuScene& s, std::size_t i) : scene(s), replica(i), receiveIndex(0) {}

void RayCallback::fetch(void *p) {
  // Only copy data if a callback was registered. If there is no call
  // back rays can still be read from DRAM when the IPU is finished.
  if (scene.getRayCallback()) {
    const auto batchIndex = receiveIndex + replica;
    ipu_utils::logger()->debug("Saving ray data from replica {} to index {}: {} bytes", replica, batchIndex, scene.getRayStreamSize());
    auto& batch = scene.getRayBatches()[batchIndex];
    std::memcpy(batch.data(), p, scene.getRayStreamSize());
    (*scene.getRayCallback())(batchIndex, batch);
  }
  // Increment the index even if no data was copied so progress can be logged:
  receiveIndex += scene.getRuntimeConfig().numReplicas;
  // Only log progress on one replica to reduce output:
  if (replica == scene.getRuntimeConfig().numReplicas - 1) {
    ipu_utils::logger()->info("Ray-batches finished: {}/{}", receiveIndex, scene.getRayBatches().size());
  }
}

poplar::StreamCallback::Result RayCallback::prefetch(void* p) {
  return poplar::StreamCallback::Result::Success;
}

void RayCallback::complete() {}

void RayCallback::invalidatePrefetched() {}
