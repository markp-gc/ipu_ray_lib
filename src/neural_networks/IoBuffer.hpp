// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <vector>

struct IoBuffer {
  IoBuffer(std::size_t batchSize, std::size_t sampleSize, std::size_t sampleCount);
  virtual ~IoBuffer() {}

  bool checkIndex();
  bool prepareNextBatchInput();
  bool storeBatchOutput();

  std::size_t batchSize;
  std::vector<float> connectedBuffer;
  std::vector<std::vector<float>> data;
  std::size_t index;
};
