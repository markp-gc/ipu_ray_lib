// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "IoBuffer.hpp"

#include <ipu_utils.hpp>
#include <sstream>

IoBuffer::IoBuffer(std::size_t batchSize, std::size_t sampleSize, std::size_t sampleCount)
:
  batchSize(batchSize),
  connectedBuffer(batchSize * sampleSize),
  data(sampleCount, std::vector<float>(sampleSize)),
  index(0u)
{}

bool IoBuffer::checkIndex() {
  if (index >= data.size()) {
    ipu_utils::logger()->debug("IoBuffer: End of buffered data at batch index {}", index);
    return false;
  }

  const auto samplesPerBatch = connectedBuffer.size() / data[index].size();
  if (samplesPerBatch != batchSize) {
    std::stringstream ss;
    ss << "IoBuffer: stream buffer and data buffer size mismatch at index "
        << index << " (samples: " << samplesPerBatch << " batch-size: " << batchSize << ")";
    throw std::runtime_error(ss.str());
  }

  return true;
}

bool IoBuffer::prepareNextBatchInput() {
  const auto startIndex = index;
  if (!checkIndex()) { return false; }

  auto bufferItr = connectedBuffer.begin();
  for (auto i = 0u; i < batchSize; ++i, ++index) {
    if (index >= data.size()) {
      ipu_utils::logger()->warn("IoBuffer: Input batch truncated at sample index {} batch index {}", i, index);
      return false;
    }
    bufferItr = std::copy(
      data[index].begin(),
      data[index].end(),
      bufferItr
    );
  }

  ipu_utils::logger()->trace("IoBuffer: Prepared input batch of samples {} to {}", startIndex, index - 1);

  return true;
}

bool IoBuffer::storeBatchOutput() {
  const auto startIndex = index;
  if (!checkIndex()) { return false; }

  auto bufferItr = connectedBuffer.begin();
  for (auto i = 0u; i < batchSize; ++i, ++index) {
    if (index >= data.size()) {
      ipu_utils::logger()->warn("IoBuffer: Output truncated at sample index {} batch index {}", i, index);
      return false;
    }
    const auto endItr = bufferItr + data[index].size();
    std::copy(bufferItr, endItr, data[index].begin());
    bufferItr = endItr;
  }

  ipu_utils::logger()->trace("IoBuffer: Retrieved output for samples {} to {}", startIndex, index - 1);
  return true;
}
