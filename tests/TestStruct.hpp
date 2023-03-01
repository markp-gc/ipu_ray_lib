// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdint>

#include <CompactBVH2Node.hpp>
#include <serialisation/Deserialiser.hpp>

#ifndef __POPC__
#include <serialisation/Serialiser.hpp>
#endif

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

struct __attribute__ ((aligned (16))) BigAlign {
  float f;
  bool b;
};

bool operator == (const BigAlign& a, const BigAlign& b) {
  return a.f == b.f && a.b == b.b;
}

#ifndef __POPC__
std::ostream& operator << (std::ostream& os, const BigAlign& a) {
  os << a.f << ", " << a.b;
  return os;
}

template <std::uint32_t BaseAlign>
void serialise(Serialiser<BaseAlign>& s, const TestStruct& t) {
  s << t.x << t.y;
  s << t.c;
  s << t.k;
  s << t.i << t.j;
}
#endif

template <std::uint32_t BaseAlign>
void deserialise(Deserialiser<BaseAlign>& d, TestStruct& t) {
  d >> t.x >> t.y >> t.c;
  d >> t.k >> t.i >> t.j;
}

#define FLOAT_TEST_DATA {1.f, 2.f, 3.f, 4.f, 5.f}

CompactBVH2Node makeTestBVHNode() {
  return CompactBVH2Node{
    1.f, 2.f, std::numeric_limits<float>::infinity(),
    123,
    (half)5.f, (half)10.f, (half)20.f,
    13
  };
}
