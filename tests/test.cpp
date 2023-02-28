// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

// This file contains all automated tests.

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <numeric>
#include <iostream>
#include <stdlib.h>

#include <CompactBVH2Node.hpp>
#include <Arrays.hpp>
#include <serialisation/Serialiser.hpp>
#include <serialisation/Deserialiser.hpp>

#include <ipu_utils.hpp>

#include "TestStruct.hpp"
#include "serialisation.hpp"
#include "deserialisation.hpp"

template <typename PODType>
struct TestExpectedPadding {
  static constexpr std::size_t value = 0;
};

template <>
struct TestExpectedPadding<std::int32_t> {
  static constexpr std::size_t value = 2;
};

template <>
struct TestExpectedPadding<float> {
  static constexpr std::size_t value = 2;
};

template <typename PODType>
void testBasicType(std::size_t bufferInitSize) {
  {
    // Create serialiser with default base alignment:
    Serialiser s(bufferInitSize);
    PODType f = (PODType)42;
    auto bytesWritten = s.write(f);
    // Default base alignment is large enough that we expect no padding for any type:
    BOOST_CHECK_EQUAL(bytesWritten, sizeof(PODType));
    // Check the values match:
    PODType ff;
    std::memcpy(&ff, s.bytes.data(), sizeof(PODType));
    BOOST_CHECK_EQUAL(f, ff);
  }
  {
    // Create serialiser with alignment 2:
    Serialiser<2> s(1024);
    PODType f = (PODType)7;
    auto bytesWritten = s.write(f);
    // How much padding do we expect?:
    constexpr auto pad = TestExpectedPadding<PODType>::value;
    BOOST_CHECK_EQUAL(bytesWritten, pad + sizeof(PODType));
    // Check the values match:
    PODType ff;
    std::memcpy(&ff, s.bytes.data() + pad, sizeof(PODType));
    BOOST_CHECK_EQUAL(f, ff);
  }
}

BOOST_AUTO_TEST_CASE(SerialiserAlign) {
  Serialiser<> s1(64);
  BOOST_CHECK_EQUAL(s1.baseAlignment(), 16);
  BOOST_CHECK_EQUAL((long)s1.bytes.data() % s1.baseAlignment(), 0);
  Serialiser<2> s2(74);
  BOOST_CHECK_EQUAL(s2.baseAlignment(), 2);
  BOOST_CHECK_EQUAL((long)s2.bytes.data() % s2.baseAlignment(), 0);
  Serialiser<1> s3(3);
  BOOST_CHECK_EQUAL(s3.baseAlignment(), 1);
  BOOST_CHECK_EQUAL((long)s3.bytes.data() % s3.baseAlignment(), 0);
}

BOOST_AUTO_TEST_CASE(SerialiseBasicTypes) {
  testBasicType<float>(1024);
  testBasicType<half>(1024);
  testBasicType<std::int32_t>(1024);
  testBasicType<std::uint16_t>(1024);
  testBasicType<std::int8_t>(1024);
}

template <std::size_t BaseAlign>
void testStruct() {
  // Test serialising a struct of POD data:
  auto p = TestStruct::TestData();

  // Initialise a serialiser with a buffer too small
  // to test reallocation during serialisation:
  Serialiser<BaseAlign> s(8);

  // Serialise the struct:
  s << p;

  // Deserialise the struct:
  TestStruct pp {0, 0, 0, 0, 0, 0};
  Deserialiser<BaseAlign> d(s.bytes);
  d >> pp;
  BOOST_CHECK_EQUAL(p.x, pp.x);
  BOOST_CHECK_EQUAL(p.y, pp.y);
  BOOST_CHECK_EQUAL(p.c, pp.c);
  BOOST_CHECK_EQUAL(p.k, pp.k);
  BOOST_CHECK_EQUAL(p.s, pp.s); // Not guaranteed by C++ standard but in reality can never break
  BOOST_CHECK_EQUAL(p.i, pp.i);
  BOOST_CHECK_EQUAL(p.j, pp.j);
}

BOOST_AUTO_TEST_CASE(SerialisePOD) {
  // Test struct serialisation with a few different alignments:
  testStruct<1>();
  testStruct<2>();
  testStruct<4>();
  testStruct<8>();
  testStruct<16>();
}

template <std::uint32_t BaseAlign>
void testCompactBvhNode() {
  CompactBVH2Node in {1.f, 2.f, std::numeric_limits<float>::infinity(),
                      123,
                      (half)5.f, (half)10.f, (half)20.f,
                      13};
  Serialiser<BaseAlign> s(0);
  s << in;
  if constexpr (BaseAlign >= 16) {
    // If byte alignment is greater or equal to 16 then the byte
    //representation should remain compact after serialisation:
    BOOST_CHECK_EQUAL(s.bytes.size(), sizeof(CompactBVH2Node));
  }

  // Check deserialise:
  Deserialiser<BaseAlign> d(s.bytes);
  CompactBVH2Node out {0, 0, 0, 0, (half)0, (half)0, (half)0, 0};
  d >> out;
  BOOST_CHECK_EQUAL(in.min_x, out.min_x);
  BOOST_CHECK_EQUAL(in.min_y, out.min_y);
  BOOST_CHECK_EQUAL(in.min_z, out.min_z);
  BOOST_CHECK_EQUAL(in.primID, out.primID);
  BOOST_CHECK_EQUAL(in.dx, out.dx);
  BOOST_CHECK_EQUAL(in.dy, out.dy);
  BOOST_CHECK_EQUAL(in.dz, out.dz);
  BOOST_CHECK_EQUAL(in.geomID, out.geomID);
}

BOOST_AUTO_TEST_CASE(SerialiseBVHNode) {
  // Test a range of alignments:
  testCompactBvhNode<1>();
  testCompactBvhNode<2>();
  testCompactBvhNode<4>();
  testCompactBvhNode<8>();
  testCompactBvhNode<16>();
}

BOOST_AUTO_TEST_CASE(SerialiseVector) {
  std::vector<std::uint32_t> v(999);
  std::iota(v.begin(), v.end(), 0);

  Serialiser<16> s(128);
  s << v;
  float f = 32.f;
  s << f;

  std::vector<std::uint32_t> out;
  Deserialiser<16> d(s.bytes);
  d >> out;
  float ff = 0.f;
  d >> ff;

  BOOST_CHECK_EQUAL(v.size(), out.size());
  for (auto i = 0u; i < v.size(); ++i) {
    BOOST_CHECK_EQUAL(v[i], out[i]);
  }
  BOOST_CHECK_EQUAL(f, ff);

  // In-place deserialisation to an array ref:
  Deserialiser<16> d2(s.bytes);
  auto outRef = deserialiseArrayRef<std::uint32_t>(d2);
  ff = 0.f;
  d2 >> ff;

  BOOST_CHECK_EQUAL(v.size(), outRef.size());
  for (auto i = 0u; i < outRef.size(); ++i) {
    BOOST_CHECK_EQUAL(v[i], outRef[i]);
  }
  BOOST_CHECK_EQUAL(f, ff);
}

ipu_utils::RuntimeConfig testConfig {
  1, 1, // numIpus, numReplicas
  "ipu_test", // exeName
  false, false, false, // useIpuModel, saveExe, loadExe
  false, true // compileOnly, deferredAttach
};

BOOST_AUTO_TEST_CASE(IpuDeserialise) {
  // Serialise some data:
  char start = 's';
  std::vector<float> array = {1.f, 2.f, 3.f, 4.f, 5.f};
  TestStruct p {1.f, 2.f, 250, 1024u, -212, +1};
  char end = 'e';

  Serialiser<16> s(1024);
  s << start;
  s << array;
  s << p;
  s << end;

  // Make a Poplar program to test it:
  using namespace poplar;
  ipu_utils::StreamableTensor bytesOnIpu("bytes");

  auto ipuTest = ipu_utils::LambdaBuilder(
    // Build test graph:
    [&](Graph& graph, const Target& target, ipu_utils::ProgramManager& progs) {
      system("popc ../tests/codelets.cpp -O3 --target ipu2 -I../include/ -I../ -I../tests -o test_codelets.gp");
      graph.addCodelets("./test_codelets.gp");

      bytesOnIpu = graph.addVariable(UNSIGNED_CHAR, {s.bytes.size()}, "byte_stream");
      graph.setTileMapping(bytesOnIpu, 0);

      auto cs1 = graph.addComputeSet("test_cs");
      auto v1 = graph.addVertex(cs1, "TestDeserialise<16>");
      graph.setTileMapping(v1, 0);

      graph.connect(v1["bytes"], bytesOnIpu);

      program::Sequence upload({
        bytesOnIpu.buildWrite(graph, true),
        program::Execute(cs1)
      });

      progs.add("test", upload);
    },
    // Run test graph:
    [&](Engine& engine, const Device& device, const ipu_utils::ProgramManager& progs) {
      bytesOnIpu.connectWriteStream(engine, s.bytes.data());
      progs.run(engine, "test");
    }
  );

  ipuTest.setRuntimeConfig(testConfig);

  auto exitCode = ipu_utils::GraphManager().run(ipuTest);
  BOOST_CHECK_EQUAL(exitCode, EXIT_SUCCESS);
}
