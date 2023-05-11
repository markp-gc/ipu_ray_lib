#include "Hdf5Model.hpp"

#include <ipu_utils.hpp>
#include <io_utils.hpp>

Hdf5Model::~Hdf5Model() {}

std::vector<Hdf5Model::JsonLayer> parseJsonModel(const std::string& s) {
  ipu_utils::logger()->trace("Model config: {}", s);
  std::stringstream ss;
  ss << s;
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(ss, pt);

  // Check its a functional model:
  auto cn = pt.get<std::string>("class_name");
  if (cn != "Functional") {
    throw std::runtime_error("Expdected a Keras 'Functional' Model");
  }

  std::vector<Hdf5Model::JsonLayer> sequential;
  auto layerTree = pt.get_child("config").get_child("layers");
  for (auto& p : layerTree) {
    cn =  p.second.get<std::string>("class_name");
    if (cn == "Dense") {
      auto cfgTree = p.second.get_child("config");
      Hdf5Model::JsonLayer layer {
        cfgTree.get<std::string>("name"),
        cfgTree.get<std::string>("activation"),
        cfgTree.get<std::string>("dtype"),
        cfgTree.get<std::size_t>("units"),
        cfgTree.get<bool>("use_bias"),
        Hdf5Model::Data(),
        Hdf5Model::Data()
      };
      ipu_utils::logger()->debug("Layer: {} {} (act: {} bias: {})", cn, layer.name, layer.activation, layer.useBias);
      sequential.push_back(layer);
    } else {
      // We ignore some layers because we hard-code their implementation in
      // Poplar to load the NIF family of models only:
      if (cn == "InputLayer" || cn == "Concatenate") {
        ipu_utils::logger()->warn("Ignoring layer (classname: {})", cn);
      } else {
        // Other than that we we only support Dense layers so throw
        // an error for any other type of layer:
        std::stringstream ss;
        ss << "Layer class: '" << cn << "' not supported by Hdf5Model loader.";
        ipu_utils::logger()->error(ss.str());
        throw std::runtime_error(ss.str());
      }
    }
  }

  return sequential;
}

const std::vector<Hdf5Model::JsonLayer>& Hdf5Model::get() const {
  return sequential;
}

Hdf5Model::Hdf5Model(const std::string& file)
  : hdf(file, H5F_ACC_RDONLY)
{
  ipu_utils::logger()->info("Reading weights saved from '{}', keras_version {}, backend {}",
                            file,
                            readStringAttribute("keras_version"),
                            readStringAttribute("backend"));
  auto jsonModelConf = readStringAttribute("model_config");
  sequential = parseJsonModel(jsonModelConf);

  for (auto& l : sequential) {
    const auto wpath = "/model_weights/" + l.name + "/" + l.name + "/kernel:0";
    ipu_utils::logger()->trace("Loading kernel data {}", wpath);
    H5::DataSet ds = hdf.openDataSet(wpath);
    l.kernelData = Data(ds);
    if (l.useBias) {
      const auto bpath = "/model_weights/" + l.name + "/" + l.name + "/bias:0";
      ipu_utils::logger()->trace("Loading bias data {}", bpath);
      H5::DataSet ds = hdf.openDataSet(bpath);
      l.biasData = Data(ds);
    }
  }

  ipu_utils::logger()->info("Finished reading model description");
}

H5std_string Hdf5Model::readStringAttribute(const std::string& attrName) {
  H5std_string str;
  auto attr = hdf.openAttribute(attrName);
  attr.read(attr.getDataType(), str);
  return str;
}

Hdf5Model::Data::Data() : numElements(0) {}

Hdf5Model::Data::Data(H5::DataSet& dset)
: shape(dset.getSpace().getSimpleExtentNdims()),
  numElements(dset.getStorageSize() / dset.getFloatType().getSize()),
  isHalfFloat(false)
{
  // NOTE: Convert everything from hsize_t to std::size_t (could be truncated):

  // Get the shape:
  std::vector<hsize_t> tmp(shape.size());
  dset.getSpace().getSimpleExtentDims(tmp.data());
  std::copy(tmp.begin(), tmp.end(), shape.begin());
  ipu_utils::logger()->trace("Data shape: {} elements: {} storage size: {}", shape, numElements, dset.getStorageSize());

  auto floatBytes = dset.getFloatType().getSize();
  ipu_utils::logger()->trace("Float size in bytes: {}", floatBytes);
  if (floatBytes == 2) {
    isHalfFloat = true;
  } else if (floatBytes == 4) {
    isHalfFloat = false;
  } else {
    throw std::runtime_error("Only float32 and float16 weights are supported.");
  }

  // Get the data values:
  ipu_utils::logger()->trace("Data num elements: {}", numElements);
  storage.resize(dset.getStorageSize());
  if (isHalf()) {
    H5::FloatType halfType;
    halfType.copy(H5::PredType::IEEE_F32BE);
    halfType.setFields(15, 10, 5, 0, 10);
    halfType.setSize(2);
    halfType.setEbias(15);
    halfType.setOrder(H5T_ORDER_LE);
    dset.read(storage.data(), halfType);
  } else {
    dset.read(storage.data(), H5::PredType::NATIVE_FLOAT);
  }
}

