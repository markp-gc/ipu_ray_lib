// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "NifMetaData.hpp"

#include <sstream>
#include <fstream>

#include <ipu_utils.hpp>
#include <io_utils.hpp>

NifMetaData::NifMetaData(const std::string& file) {
  std::ifstream stream(file);
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(stream, pt);
  if (pt.empty()) {
    throw std::runtime_error("Empty property tree after parsing file: '" + file + "'");
  }

  try {
    embeddingDimension = pt.get<std::size_t>("embedding_dimension");
    name = pt.get<std::string>("name");

    for (auto& p: pt) {
      ipu_utils::logger()->trace("Found property: {}", p.first);
    }

    auto shapeTree = pt.get_child("original_image_shape");
    for (auto& p: shapeTree) {
      imageShape.push_back(std::atoi(p.second.data().c_str()));
    }

    auto encoderTree = pt.get_child("encode_params");
    eps = encoderTree.get<float>("eps");
    logToneMap = encoderTree.get<bool>("log_tone_map");
    max = encoderTree.get<float>("max");
    ipu_utils::logger()->debug("Encoder eps: {}", eps);
    ipu_utils::logger()->debug("Encoder log tone-mapping: {}", logToneMap);
    ipu_utils::logger()->debug("Encoder max: {}", max);
    auto meanTree = encoderTree.get_child("mean");

    mean.clear();
    mean.reserve(3);
    for (auto& p: meanTree) {
      mean.push_back(p.second.get_value<float>());
    }
    ipu_utils::logger()->debug("Encoder mean: {}", mean);

    // If tone-mapping fold the inverse eps into the mean:
    if (logToneMap) {
      mean[0] -= eps;
      mean[1] -= eps;
      mean[2] -= eps;
    }

    // TODO should get this from keras model itself:
    auto cmdTree = pt.get_child("train_command");
    bool next = false;
    for (auto& p: cmdTree) {
      if (next) {
        hiddenSize = std::atoi(p.second.data().c_str());
        next = false;
      }
      if (p.second.data() == "--layer-size") { next = true; }
    }

  } catch (const std::exception& e) {
    std::stringstream ss;
    ss << "Error reading property: " << e.what() << " from file: '" << file << "'";
    throw std::runtime_error(ss.str());
  }
}
