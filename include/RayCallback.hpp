// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <poplar/StreamCallback.hpp>

class IpuScene;

// Callback for connection to an output stream that
// receives each batch of partial results from the IPU.
class RayCallback : public poplar::StreamCallback {
public:
  RayCallback(IpuScene& s, std::size_t i);
  void fetch(void *p);
  poplar::StreamCallback::Result prefetch(void* p);
  void complete();
  void invalidatePrefetched();

private:
  IpuScene& scene;
  const std::size_t replica;
  std::size_t receiveIndex;
};

