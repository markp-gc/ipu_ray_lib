// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

// Functions for deserialising commonly used types:

#include <serialisation/Deserialiser.hpp>
#include <CompactBVH2Node.hpp>
#include <Arrays.hpp>

template <std::uint32_t BaseAlign>
void deserialise(Deserialiser<BaseAlign>& d, CompactBVH2Node& n) {
  d >> n.min_x >> n.min_y >> n.min_z;
  d >> n.primID;
  d >> n.dx >> n.dy >> n.dz;
  d >> n.geomID;
}

#ifndef __POPC__
template <std::uint32_t BaseAlign, typename T>
void deserialise(Deserialiser<BaseAlign>& d, std::vector<T>& arr) {
  std::uint64_t size;
  d >> size;
  arr.clear();
  arr.resize(size);
  d.read((std::uint8_t*)arr.data(), size * sizeof(T));
}
#endif

template <typename T, std::uint32_t BaseAlign>
ConstArrayRef<T> deserialiseArrayRef(Deserialiser<BaseAlign>& d) {
  std::uint64_t size;
  d >> size;
  ConstArrayRef<T> arr((T*)d.getPtr(), size);
  d.skip(size * sizeof(T));
  return arr;
}
