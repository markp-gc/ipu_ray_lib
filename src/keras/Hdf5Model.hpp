#pragma once

#include <H5Cpp.h>

#include <vector>
#include <map>

using TensorShape = std::vector<std::size_t>;

// Very limited H5 model reading (Many hard coded aspects so
// only useable for a known Keras model).
struct Hdf5Model {

  struct Data {
    Data();
    Data(H5::DataSet& dset);

    std::size_t rank() const { return shape.size(); }
    std::size_t elements() const { return numElements; }
    bool isHalf() const { return isHalfFloat; }
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> storage;

  private:
    std::size_t numElements;
    bool isHalfFloat;
  };

  struct JsonLayer {
    std::string name;
    std::string activation;
    std::string dtype;
    std::size_t units;
    bool useBias;
    Data kernelData;
    Data biasData;
  };

  Hdf5Model(const std::string& file);
  virtual ~Hdf5Model();

  H5std_string readStringAttribute(const std::string& attrName);

  const std::vector<JsonLayer>& get() const;

private:
  H5::H5File hdf;
  std::vector<JsonLayer> sequential;
};
