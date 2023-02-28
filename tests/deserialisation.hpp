// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

// Fuctions for deserialising data used in tests

#include <CompactBVH2Node.hpp>
#include <serialisation/Deserialiser.hpp>
#include "TestStruct.hpp"

template <std::uint32_t BaseAlign>
void deserialise(Deserialiser<BaseAlign>& d, TestStruct& t) {
  d >> t.x >> t.y >> t.c;
  d >> t.k >> t.i >> t.j;
}

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
  arr.reserve(size);
  for (auto i = 0u; i < size; ++i) {
    T v;
    d >> v;
    arr.push_back(v);
  }
}
#endif

template <typename T, std::uint32_t BaseAlign>
ArrayRef<T> deserialiseArrayRef(Deserialiser<BaseAlign>& d) {
  std::uint64_t size;
  d >> size;
  ArrayRef<T> arr((T*)d.getPtr(), size);
  d.skip(size * sizeof(T));
  return arr;
}
