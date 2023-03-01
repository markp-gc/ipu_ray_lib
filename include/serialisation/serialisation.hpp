// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

// Functions for serialising commonly used types:

#include <serialisation/Serialiser.hpp>
#include <Arrays.hpp>

template <std::uint32_t BaseAlign>
void serialise(Serialiser<BaseAlign>& s, const CompactBVH2Node& n) {
  s << n.min_x << n.min_y << n.min_z;
  s << n.primID;
  s << n.dx << n.dy << n.dz;
  s << n.geomID;
}

template <std::uint32_t BaseAlign, typename T>
void serialise(Serialiser<BaseAlign>& s, const std::vector<T>& arr) {
  std::uint64_t size = arr.size(); // size_t is not portable
  s << size;
  s.write((const T*)arr.data(), size);
}

template <std::uint32_t BaseAlign, typename T>
void serialise(Serialiser<BaseAlign>& s, const ConstArrayRef<T>& arr) {
  std::uint64_t size = arr.size(); // size_t is not portable
  s << size;
  s.write(arr.cbegin(), size);
}
