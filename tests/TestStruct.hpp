// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdint>

#include <CompactBVH2Node.hpp>

struct
__attribute__ ((aligned (8)))
TestStruct {
  float x, y;
  std::uint8_t c; // char here to mess up alignment
  union {
    std::uint16_t k;
    half s;
  };
  std::int32_t i, j;

  static TestStruct TestData() {
    return TestStruct{1.f, 2.f, 250, 1024u, -212, +1};
  }
};

#define FLOAT_TEST_DATA {1.f, 2.f, 3.f, 4.f, 5.f}

CompactBVH2Node makeTestBVHNode() {
  return CompactBVH2Node{
    1.f, 2.f, std::numeric_limits<float>::infinity(),
    123,
    (half)5.f, (half)10.f, (half)20.f,
    13
  };
}
