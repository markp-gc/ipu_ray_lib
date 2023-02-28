// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

// This file contains IPU compute codelets (kernels) for the various test programs

#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <array>
#include <print.h>
#include <assert.h>

#include <serialisation/Deserialiser.hpp>
#include <Arrays.hpp>

#include "TestStruct.hpp"
#include "deserialisation.hpp"

template <typename T>
void CHECK_EQUAL(const T& a, const T& b) {
  if (a != b) {
    printf("CHECK_EQUAL(a, b) FAILED\n");
  }
  assert(a == b);
}

using namespace poplar;

template <std::size_t SerialAlign>
class TestDeserialise : public Vertex {
public:
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, SerialAlign>> bytes;

  bool compute() {
    // inputs:
    char start = 's';
    std::array<float, 5> array = {1.f, 2.f, 3.f, 4.f, 5.f};
    auto p = TestStruct::TestData();
    char end = 'e';

    // outputs:
    char dstart = 's';
    TestStruct pp {0, 0, 0, 0, 0, 0};
    char dend = 'e';
    Deserialiser<SerialAlign> d((std::uint8_t*)&bytes[0], bytes.size());
    d >> dstart;
    auto darray = deserialiseArrayRef<float>(d);
    d >> pp;
    d >> dend;

    CHECK_EQUAL(start, dstart);
    CHECK_EQUAL(end, dend);
    CHECK_EQUAL(p.x, pp.x);
    CHECK_EQUAL(p.y, pp.y);
    CHECK_EQUAL(p.c, pp.c);
    CHECK_EQUAL(p.k, pp.k);
    CHECK_EQUAL(p.s, pp.s);
    CHECK_EQUAL(p.i, pp.i);
    CHECK_EQUAL(p.j, pp.j);

    for (auto i = 0u; i < array.size(); ++i) {
      CHECK_EQUAL(array[i], darray[i]);
    }

    return true;
  }
};

template class TestDeserialise<16>;
