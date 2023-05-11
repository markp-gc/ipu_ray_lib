// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <vector>
#include <string>

using TensorShape = std::vector<std::size_t>;

struct NifMetaData {
  NifMetaData(const std::string& file);
  virtual ~NifMetaData() {}

  std::string name;
  std::size_t embeddingDimension;
  std::size_t hiddenSize;
  TensorShape imageShape;

  // Encoder params:
  std::vector<float> mean;
  float eps;
  float max;
  bool logToneMap;
};
