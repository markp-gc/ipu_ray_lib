// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "NifModel.hpp"

#include "IoBuffer.hpp"

#include <poplar/VariableMappingMethod.hpp>
#include <poplar/CycleCount.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>

#include <algorithm>

#include <io_utils.hpp>


namespace {

std::size_t getFirstTile(poplar::Graph& g, poplar::Tensor t) {
  if (!t.valid()) {
    throw std::runtime_error("Un-initialised poplar::Tensor.");
  }

  auto m = g.getTileMapping(t);
  for (auto i = 0u; i < m.size(); ++i) {
    if (!m[i].empty()) {
      return i;
    }
  }

  throw std::runtime_error("Tensor '" + t.getDebugStr() + "' has no tile mapping in this graph.");
}

} // end anonymous namespace

NifModel::Data::Data(const std::string& h5File, const std::string& metaFile)
: metaData(metaFile)
{
  ipu_utils::logger()->info("Loading model metadata from file: '{}'", metaFile);
  ipu_utils::logger()->debug("Loaded NIF metadata for model name: {}", metaData.name);
  ipu_utils::logger()->debug("NIF embedding dimension: {}", metaData.embeddingDimension);
  ipu_utils::logger()->debug("NIF hidden dimension: {}", metaData.hiddenSize);
  ipu_utils::logger()->debug("Reconstructed image shape: {}", metaData.imageShape);
  setupModel(h5File);
}

void NifModel::Data::setupModel(const std::string& h5File) {

  // TODO: model loader is hard coded for sequential dense only layers:
  Hdf5Model h5model(h5File);

  auto i = 0u;
  for (auto& l : h5model.get()) {
    auto dtype = l.dtype == "float16" ? poplar::HALF : poplar::FLOAT;
    layers.emplace_back(l.kernelData.shape, dtype, l.activation, l.name);
    auto& newLayer = layers.back();
    newLayer.kernel.data = l.kernelData.storage;
    ipu_utils::logger()->debug("Added dense kernel: {} size: {}", newLayer.kernel.getName(), newLayer.kernel.data.size());
    if (l.useBias) {
      newLayer.bias.data = l.biasData.storage;
      ipu_utils::logger()->debug("Added bias: {} size: {}", newLayer.bias.getName(), newLayer.bias.data.size());
      ipu_utils::logger()->debug("Layer {}: weight tensors: {} ({}) {} ({})",
        i, newLayer.kernel.getName(), newLayer.kernel.shape, newLayer.bias.getName(), newLayer.bias.shape);
    } else {
      ipu_utils::logger()->debug("Layer {}: weight tensors: {} ({})",
        i, newLayer.kernel.getName(), newLayer.kernel.shape);
    }

    // Rename linear -> none
    if (newLayer.activationFunction == "linear") {
      newLayer.activationFunction = "none";
    }

    // Sanity check values look ok:
    if (newLayer.kernel.type == poplar::FLOAT) {
      ipu_utils::logger()->trace("Kernel first value: {}", ((float*)newLayer.kernel.data.data())[0]);
    }

    i += 1;
  }
}

NifModel::NifModel(std::shared_ptr<Data>& sharedData, const std::string& modelName)
: data(sharedData),
  name(modelName),
  batchSize(0u),
  max(modelName + "/max"),
  mean(modelName + "/mean"),
  input(modelName + "/input"),
  output(modelName + "/output"),
  cycleCount(modelName + "/cycle_count"),
  cycleCountResult(std::numeric_limits<std::uint64_t>::max()),
  inferenceBuilt(false),
  streamedIO(false),
  inputU(modelName + "/inputU"),
  inputV(modelName + "/inputV"),
  decodeOnDevice(true)
{
  setupStreamableTensors();
}

NifModel::NifModel(std::shared_ptr<Data>& sharedData, const std::string& modelName, bool deviceDecoder, std::size_t batchSizeOverride)
  : NifModel(sharedData, modelName)
{
  decodeOnDevice = deviceDecoder;
  if (batchSizeOverride == 0u) {
    batchSize = *std::max_element(data->getMetaData().imageShape.begin(), data->getMetaData().imageShape.end());
    ipu_utils::logger()->debug("Auto selected batch-size: {}", batchSize);
  } else {
    batchSize = batchSizeOverride;
    ipu_utils::logger()->debug("Forced batch-size: {}", batchSize);
  }

  setupIoBuffers();
}

/// Calculate and log useful information about the model:
void NifModel::analyseModel(std::size_t sampleCount) const {
  std::size_t flops = 0;
  std::size_t parametersBytes = 0;

  for (const auto &l : data->getLayers()) {
    parametersBytes += l.kernel.data.size() + l.bias.data.size();

    auto layerFlops = 2 * l.kernel.shape[0] * l.kernel.shape[1];
    if (l.hasBias()) {
      layerFlops += l.bias.shape[0];
    }
    flops += layerFlops;
  }

  auto parametersKiB = parametersBytes / 1024.f;
  flops *= sampleCount;

  ipu_utils::logger()->info("NIF {} layers: {}", name, data->getLayers().size());
  ipu_utils::logger()->info("NIF {} Hidden size: {}", name, data->getLayers().front().kernel.shape[1]);
  ipu_utils::logger()->info("NIF {} batch size: {}", name, sampleCount);
  ipu_utils::logger()->info("NIF {} model FLOPS: {}", name, flops);
  ipu_utils::logger()->info("NIF {} parameter size: {} KiB", name, parametersKiB);
}

void insertStreamableTensorChecked(
  const std::string& key,
  const std::string& modelName,
  std::map<std::string, ipu_utils::StreamableTensor>& store)
{
  if (store.count(key)) {
    throw std::logic_error("Streamable tensor with this name already added.");
  }
  auto streamableName = modelName + "/" + key;
  store.insert(std::make_pair(key, ipu_utils::StreamableTensor(streamableName)));
}

// Need to create a streamable tensor
// for each layer in the model:
void NifModel::setupStreamableTensors() {
  for (const auto& l : data->getLayers()) {
    std::string key = l.kernel.getName();
    insertStreamableTensorChecked(key, name, modelTensors);
    if (l.hasBias()) {
      key = l.bias.getName();
      insertStreamableTensorChecked(key, name, modelTensors);
    }
  }
}

void NifModel::setupIoBuffers() {
  auto sampleCount = data->getMetaData().imageShape[0] * data->getMetaData().imageShape[1];
  inputBufferU = std::make_unique<IoBuffer>(batchSize, 1, sampleCount);
  inputBufferV = std::make_unique<IoBuffer>(batchSize, 1, sampleCount);
  inputBuffer = std::make_unique<IoBuffer>(batchSize, data->getLayers().front().kernel.shape.front(), sampleCount);
  outputBuffer = std::make_unique<IoBuffer>(batchSize, data->getLayers().back().kernel.shape.back(), sampleCount);

  ipu_utils::logger()->debug("Output stream buffer size: {}", outputBuffer->connectedBuffer.size());
  ipu_utils::logger()->debug("NifModel '{}': Total output data: {} x {}", name, outputBuffer->data.size(), outputBuffer->data.back().size());
}

NifModel::~NifModel() {}

/// Build the input encoding program (generate Fourier features from UV coords):
poplar::Tensor NifModel::buildEncodeInput(poplar::Graph& g, poplar::Type dtype, poplar::Tensor uvCoords, poplar::program::Sequence& prog) {
  std::string opPrefix = name + "/input_encoding";

  // Compute powers on host and upload as constant. This avoids using powf on
  // device which is slow and wastes memory with double emulation code:
  auto powers = makeCoefficients();
  auto coeffs = g.addConstant(poplar::FLOAT, {data->getMetaData().embeddingDimension}, powers.data(), opPrefix + "/powers");
  auto firstInputTile = getFirstTile(g, uvCoords);
  g.setTileMapping(coeffs, firstInputTile);

  auto one = g.addConstant(poplar::FLOAT, {}, 1.f, opPrefix + "/one");
  auto two = g.addConstant(poplar::FLOAT, {}, 2.f, opPrefix + "/two");
  g.setTileMapping(one, firstInputTile);
  g.setTileMapping(two, firstInputTile);

  // uvNorm = 2 * (uvCoords - 1):
  namespace pe = popops::expr;
  auto normExpr = pe::Mul(pe::Sub(pe::_1, pe::_2), pe::_3);
  popops::mapInPlace(g, normExpr, {uvCoords, one, two}, prog, opPrefix + "/norm");

  auto uv = uvCoords.slice({0, 0}, {2, batchSize}).expand({2});
  coeffs = coeffs.expand({0}).broadcast(batchSize, 0).expand({0});
  auto posuv = popops::mul(g, uv, coeffs, prog, opPrefix + "/coeff_mul");

  // sin() and cos(). Do cosine first then the sine in place.
  // Cast to fp16 because fp32 implementations are currently slow:
  auto posuv_fp16 = popops::cast(g, posuv, poplar::HALF, prog, opPrefix + "/to_fp16");
  auto cosuv_fp16 = popops::cos(g, posuv_fp16, prog, opPrefix + "/cos_fp16");
  popops::sinInPlace(g, posuv_fp16, prog, opPrefix + "/sin_fp16");
  posuv = popops::cast(g, posuv_fp16, dtype, prog, opPrefix + "/to_fp32");
  auto cosuv = popops::cast(g, cosuv_fp16, dtype, prog, opPrefix + "/to_fp32");
  auto fourierFeatures = poplar::concat({posuv[0], posuv[1], cosuv[0], cosuv[1]}, 1);
  return fourierFeatures;
}

/// Build program to apply mean shift and tone-mapping. Applies in-place if possible.
poplar::Tensor NifModel::buildDecodeOutput(poplar::Graph& g, poplar::Tensor bgr, poplar::program::Sequence& prog) {
  std::string opPrefix = name + "/output_decoding";
  auto firstInputTile = getFirstTile(g, bgr);

  // Always do output decoding at fp32:
  bgr = popops::cast(g, bgr, poplar::FLOAT, prog);
  max = g.addVariable(poplar::FLOAT, {}, opPrefix + "/max");
  g.setTileMapping(max, firstInputTile);
  popops::mulInPlace(g, bgr, max.get(), prog, opPrefix + "/scale_max");

  if (data->getMetaData().logToneMap) {
    ipu_utils::logger()->info("NifModel '{}': Building log-tonemapped decoder. Compiled graph will only be suitable for HDR images.", name);
  }

  mean = g.addVariable(poplar::FLOAT, {1, 3}, name + "/mean");
  g.setTileMapping(mean, firstInputTile);

  popops::addInPlace(g, bgr, mean.get(), prog, opPrefix + "/offset_mean");

  if (data->getMetaData().logToneMap) {
    popops::expInPlace(g, bgr, prog, opPrefix + "/tonemap_exp");
  }

  return bgr;
}

/// Build the main model inference program:
poplar::program::Sequence
NifModel::buildInference(poplar::Graph& g,
      poplar::OptionFlags& matmulOptions,
      poplin::matmul::PlanningCache& cache,
      bool optimiseStreamMemory,
      poplar::Tensor inputUV) {
  popops::addCodelets(g);
  poplin::addCodelets(g);

  poplar::program::Sequence progWithIO;
  poplar::program::Sequence execModel;
  const auto dtype = poplar::FLOAT;

  if (inputUV.valid()) {
    ipu_utils::logger()->debug("{}: UV input tensor was provided with shape: {}", name, inputUV.shape());
    inputUV = inputUV.reshape({2, inputUV.numElements()/2});
    ipu_utils::logger()->debug("{}: UV input tensor reshaped to: {}", name, inputUV.shape());
    batchSize = inputUV.shape().back();
    ipu_utils::logger()->debug("{}: Batch size set to: {}", name, batchSize);
    streamedIO = false;
  } else {
    // No input tensor passed so create one and set it up for streaming:
    ipu_utils::logger()->debug("{}: No input tensor provided. Input will be allocated for stremaing.", name);
    constexpr auto linearMapping = poplar::VariableMappingMethod::LINEAR;
    inputU = g.addVariable(dtype, {batchSize}, linearMapping, name + "/inputU");
    inputV = g.addVariable(dtype, {batchSize}, linearMapping, name + "/inputV");
    progWithIO.add(inputU.buildWrite(g, optimiseStreamMemory));
    progWithIO.add(inputV.buildWrite(g, optimiseStreamMemory));
    inputUV = poplar::concat({inputU.get().expand({0}), inputV.get().expand({0})}, 0);
    streamedIO = true;
  }

  // Lay out input for first matmul:
  auto kernelShape = data->getLayers().front().kernel.shape;
  auto firstKernelType = data->getLayers().front().kernel.type;
  const TensorShape inputShape = {batchSize, kernelShape.front()};
  ipu_utils::logger()->debug("NifModel '{}': Input shape: {}", name, inputShape);

  input = poplin::createMatMulInputLHS(g, firstKernelType, dtype,
    inputShape, kernelShape, "fourier_features", matmulOptions, &cache);

  auto encoded = buildEncodeInput(g, dtype, inputUV, execModel);
  if (input.elementType() != encoded.elementType()) {
    encoded = popops::cast(g, encoded, input.elementType(), execModel, name + "/cast_encodings");
  }
  execModel.add(poplar::program::Copy(encoded, input));

  // Build core MLP model from the layer descriptions:

  // Sequence for front end of model:
  poplar::Tensor x = input;
  for (auto i = 0u; i < data->getLayers().size(); ++i) {
    auto& l = data->getLayers()[i];
    kernelShape = l.kernel.shape;

    // Auto-detect the concat point in the NIF network (once we can properly
    // load any H5 (or other) format model this hack won't be necessary):
    if (x.shape().back() != kernelShape.front()) {
      x = poplar::concat(x, input, 1);
      ipu_utils::logger()->debug("NifModel '{}': Detected network back end: acts concatted with input to give shape: {}", name, x.shape());
    }

    // Build the rhs and matmul op for the layer:
    auto& kernelTensor = modelTensors.at(l.kernel.getName());
    kernelTensor = poplin::createMatMulInputRHS(g, l.kernel.type, dtype, x.shape(), kernelShape, kernelTensor.getName(), matmulOptions, &cache);
    std::string opPrefix = name + "/layer_" + std::to_string(i) + "_";
    x = poplin::matMul(g, x, kernelTensor, execModel, l.kernel.type, opPrefix + "matmul", matmulOptions, &cache);
    // Bias if needed:
    if (l.hasBias()) {
      auto& biasTensor = modelTensors.at(l.bias.getName());
      biasTensor = g.addVariable(l.bias.type, l.bias.shape);
      g.setTileMapping(biasTensor, g.getTileMapping(x[0]));
      popops::addInPlace(g, x, biasTensor.get(), execModel, opPrefix + "add_bias");
    }

    if (l.activationFunction == "relu") {
      popnn::nonLinearityInPlace(g, popnn::NonLinearityType::RELU, x, execModel, opPrefix + "relu");
    }
  }

  if (decodeOnDevice) {
    output = buildDecodeOutput(g, x, execModel);
  } else {
    output = x;
  }

  inferenceBuilt = true;

  // Only build reads of output and cycle count if the
  // model is not being used inline in a larger program:
  if (streamedIO) {
    // Add a cycle count around the model:
    cycleCount = poplar::cycleCount(g, execModel, 0, poplar::SyncType::INTERNAL, name + "/cycle_count");

    // Execute the neural network:
    progWithIO.add(execModel);

    // Stream back results:
    progWithIO.add(output.buildRead(g, optimiseStreamMemory));
    progWithIO.add(cycleCount.buildRead(g, optimiseStreamMemory));
    ipu_utils::logger()->debug("NifModel '{}': Output shape: {}", name, output.shape());

    return progWithIO;
  }

  // Only return program that executes neural network:
  return execModel;
}

poplar::program::Sequence NifModel::buildInit(poplar::Graph& g, bool optimiseStreamMemory) {
  if (!inferenceBuilt) {
    throw std::runtime_error("You must call 'buildInference' before you call 'buildInit'.");
  }

  // Program to initialise the weights for all layers:
  poplar::program::Sequence initProg;
  for (auto& p : modelTensors) {
    initProg.add(p.second.buildWrite(g, optimiseStreamMemory));
  }

  // Also need to stream mean and max params:
  initProg.add(max.buildWrite(g, true));
  initProg.add(mean.buildWrite(g, true));

  return initProg;
}

void NifModel::connectStreams(poplar::Engine& engine) {
  if (streamedIO) {
    cycleCount.connectReadStream(engine, &cycleCountResult);

    ipu_utils::logger()->trace("NifModel '{}': Connecting output stream: ({} elements)", name, outputBuffer->connectedBuffer.size());
    output.connectReadStream(engine, outputBuffer->connectedBuffer);

    inputU.connectWriteStream(engine, inputBufferU->connectedBuffer);
    inputV.connectWriteStream(engine, inputBufferV->connectedBuffer);
  }

  // Connect tone-map-decode parameters:
  max.connectWriteStream(engine, &data->getMetaData().max);
  mean.connectWriteStream(engine, data->getMetaData().mean.data());

  // Connect layer weights:
  for (auto& l : data->getLayers()) {
    auto& kernelTensor = modelTensors.at(l.kernel.getName());
    ipu_utils::logger()->trace("NifModel '{}': Connecting weight stream: ({} bytes)", name, l.kernel.data.size());
    kernelTensor.connectWriteStream(engine, l.kernel.data);
    if (l.hasBias()) {
      auto& biasTensor = modelTensors.at(l.bias.getName());
      ipu_utils::logger()->trace("NifModel '{}': Connecting bias stream: ({} bytes)", name, l.bias.data.size());
      biasTensor.connectWriteStream(engine, l.bias.data);
    }
  }
}

/// Generate host input samples to reconstruct the whole image:
void NifModel::generateInputSamples() {
  std::vector<float> uCoords;
  std::vector<float> vCoords;
  std::tie(uCoords, vCoords) = makeGridCoordsUV();
  auto coeffs = makeCoefficients();

  // Fill the raw UV input buffer:
  for (auto i = 0u; i < uCoords.size(); ++i) {
    inputBufferU->data[i][0] = uCoords[i];
    inputBufferV->data[i][0] = vCoords[i];
  }

  // Fill an input stream positionally encoded on host:
  const auto dimBy4 = data->getMetaData().embeddingDimension;
  for (auto i = 0u; i < uCoords.size(); ++i) {
    auto& u = uCoords[i];
    auto& v = vCoords[i];
    u = 2.f * (u - 1.f);
    v = 2.f * (v - 1.f);
    auto& encoded = inputBuffer->data[i];
    for (auto j = 0u; j < dimBy4; ++j) {
      auto posx = u * coeffs[j];
      auto posy = v * coeffs[j];
      encoded[j] = std::sin(posx);
      encoded[j + dimBy4] = std::sin(posy);
      encoded[j + 2*dimBy4] = std::cos(posx);
      encoded[j + 3*dimBy4] = std::cos(posy);
    }
  }

  if (!prepareNextBatch()) {
    throw std::runtime_error("Could not prepare first batch.");
  }
}

bool NifModel::prepareNextBatch() {
  bool allOk = inputBufferU->prepareNextBatchInput();
  allOk &= inputBufferV->prepareNextBatchInput();
  return allOk;
}

bool NifModel::storeBatchOutput() {
  return outputBuffer->storeBatchOutput();
}

void NifModel::saveImage(const std::string& fileName) {
  auto height = data->getMetaData().imageShape[0];
  auto width = data->getMetaData().imageShape[1];
  const auto& samples = decodeOnDevice ? outputBuffer->data : decodeSamples();

  cv::Mat image(height, width, CV_32FC3);
  auto itr = samples.begin();
  for (auto r = 0u; r < height; r++) {
    for (auto c = 0u; c < width; ++c) {
      auto & bgr = *itr;
      image.at<cv::Vec3f>(r, c) = cv::Vec3f(bgr[0], bgr[1], bgr[2]);
      itr += 1;
    }
  }
  cv::imwrite(fileName, image);
}

std::vector<float> NifModel::makeCoefficients() {
  std::vector<float> powers(data->getMetaData().embeddingDimension);
  for (auto i = 0u; i < powers.size(); ++i) {
    powers[i] = (float)std::pow(2, i);
  }
  return powers;
}

std::pair<std::vector<float>, std::vector<float>> NifModel::makeGridCoordsUV() {
  auto width = data->getMetaData().imageShape[1];
  auto height = data->getMetaData().imageShape[0];
  ipu_utils::logger()->debug("NifModel '{}': generating uv coords for image wxh: {} x {}", name, width, height);
  std::vector<float> u(width * height);
  std::vector<float> v(width * height);
  auto i = 0u;
  for (auto r = 0u; r < height; ++r) {
    for (auto c = 0u; c < width; ++c) {
      u[i] = r / (float)height;
      v[i] = c / (float)width;
      i += 1;
    }
  }
  ipu_utils::logger()->debug("NifModel '{}': {} UV coord pairs generated", name, i);
  return std::make_pair(u, v);
}

const std::vector<std::vector<float>>& NifModel::decodeSamples() {
  for (auto& bgr : outputBuffer->data) {
    bgr[0] *= data->getMetaData().max;
    bgr[1] *= data->getMetaData().max;
    bgr[2] *= data->getMetaData().max;
    bgr[0] += data->getMetaData().mean[0];
    bgr[1] += data->getMetaData().mean[1];
    bgr[2] += data->getMetaData().mean[2];

    if (data->getMetaData().logToneMap) {
      bgr[0] = std::exp(bgr[0] - data->getMetaData().eps);
      bgr[1] = std::exp(bgr[1] - data->getMetaData().eps);
      bgr[2] = std::exp(bgr[2] - data->getMetaData().eps);
    }
  }

  return outputBuffer->data;
}
