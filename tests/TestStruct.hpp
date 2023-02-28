// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdint>

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
