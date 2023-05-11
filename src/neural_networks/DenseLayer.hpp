// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

struct HostTensor {
  HostTensor(const std::vector<std::size_t>& shape, poplar::Type dtype, const std::string& name)
    : shape(shape), nameSuffix(name), type(dtype) {}

  const std::string& getName() const { return nameSuffix; }

  std::vector<std::size_t> shape;
  std::string nameSuffix;
  //ipu_utils::StreamableTensor tensor;
  poplar::Type type;
  std::vector<std::uint8_t> data;
};

struct DenseLayer {
  DenseLayer(const std::vector<std::size_t>& shape, poplar::Type dtype, const std::string& activation, const std::string& layerName)
  :
    kernel(shape, dtype, layerName + "/kernel"),
    bias({shape.back()}, dtype, layerName + "/bias"),
    activationFunction(activation)
  {}

  bool hasBias() const { return !bias.data.empty(); }

  HostTensor kernel;
  HostTensor bias;
  std::string activationFunction;
};
