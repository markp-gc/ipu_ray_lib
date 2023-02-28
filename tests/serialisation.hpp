// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

// Fuctions for serialising data used in tests

#include <serialisation/Serialiser.hpp>

template <std::uint32_t BaseAlign>
void serialise(Serialiser<BaseAlign>& s, const TestStruct& t) {
  s << t.x << t.y;
  s << t.c;
  s << t.k;
  s << t.i << t.j;
}

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
  for (const auto& v : arr) {
    s << v;
  }
}
