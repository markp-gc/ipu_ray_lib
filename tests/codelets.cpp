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

#define CHECK_EQUAL(a,b) \
do { \
  if ((a) != (b)) { \
    printf("\n"); assert(a == b); \
  } \
} while(0)

using namespace poplar;

template <std::size_t SerialAlign>
class TestDeserialise : public Vertex {
public:
  Input<Vector<unsigned char, poplar::VectorLayout::SPAN, SerialAlign>> bytes;

  bool compute() {
    // inputs:
    char start = 's';
    std::array<float, 5> array = FLOAT_TEST_DATA;
    auto p = TestStruct::TestData();
    char end = 'e';

    // outputs:
    char dstart = 's';
    TestStruct pp {0, 0, 0, 0, 0, 0};
    CompactBVH2Node node;
    char dend = 'e';
    Deserialiser<SerialAlign> d((std::uint8_t*)&bytes[0], bytes.size());
    d >> dstart;
    auto darray = deserialiseArrayRef<float>(d);
    d >> pp;
    d >> node;
    d >> dend;

    CHECK_EQUAL(start, dstart);

    for (auto i = 0u; i < array.size(); ++i) {
      CHECK_EQUAL(array[i], darray[i]);
    }

    CHECK_EQUAL(p.x, pp.x);
    CHECK_EQUAL(p.y, pp.y);
    CHECK_EQUAL(p.c, pp.c);
    CHECK_EQUAL(p.k, pp.k);
    CHECK_EQUAL(p.s, pp.s);
    CHECK_EQUAL(p.i, pp.i);
    CHECK_EQUAL(p.j, pp.j);

    auto n = makeTestBVHNode();
    CHECK_EQUAL(n.min_x, node.min_x);
    CHECK_EQUAL(n.min_y, node.min_y);
    CHECK_EQUAL(n.min_z, node.min_z);
    CHECK_EQUAL(n.primID, node.primID);
    CHECK_EQUAL(n.dx, node.dx);
    CHECK_EQUAL(n.dy, node.dy);
    CHECK_EQUAL(n.dz, node.dz);
    CHECK_EQUAL(n.geomID, node.geomID);

    CHECK_EQUAL(end, dend);

    return true;
  }
};

template class TestDeserialise<16>;
