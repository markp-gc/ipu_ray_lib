// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

// Functions for serialising commonly used types:

#include <serialisation/Serialiser.hpp>
#include <Arrays.hpp>
#include <Primitives.hpp>
#include <Scene.hpp>

template <std::uint32_t BaseAlign>
void serialise(Serialiser<BaseAlign>& s, const CompactBVH2Node& n) {
  s << n.min_x << n.min_y << n.min_z;
  s << n.primID;
  s << n.dx << n.dy << n.dz;
  s << n.geomID;
}

template <std::uint32_t BaseAlign, typename T>
void serialise(Serialiser<BaseAlign>& s, const std::vector<T>& arr) {
  std::uint32_t size = arr.size(); // size_t is not portable
  s << size;
  s.write((const T*)arr.data(), size);
}

template <std::uint32_t BaseAlign, typename T>
void serialise(Serialiser<BaseAlign>& s, const ArrayRef<T>& arr) {
  std::uint32_t size = arr.size(); // size_t is not portable
  s << size;
  s.write(arr.cbegin(), size);
}

template <std::uint32_t BaseAlign>
void serialise(Serialiser<BaseAlign>& ss, const SceneRef& s) {
  ss << s.geometry;
  ss << s.meshInfo;
  ss << s.meshTris;
  ss << s.meshVerts;
  ss << s.meshNormals;
  ss << s.matIDs;
  ss << s.materials;
  ss << s.bvhNodes;
  ss << s.maxLeafDepth;
  ss << s.imageWidth;
  ss << s.imageHeight;
  ss << s.fovRadians;
  ss << s.antiAliasScale;
  ss << s.maxPathLength;
  ss << s.rouletteStartDepth;
  ss << s.samplesPerPixel;
}

// Specialisation for primitives and arrays of primitives
// (so that we only serialise data not vtable etc):
template <std::uint32_t BaseAlign>
void serialise(Serialiser<BaseAlign>& ss, const Sphere& s) {
  ss << s.x << s.y << s.z << s.radius;
}

template <std::uint32_t BaseAlign>
void serialise(Serialiser<BaseAlign>& ss, const Disc& d) {
  ss << d.cx << d.cy << d.cz
     << d.r
     << d.nx << d.nz << d.nz;
}

template <std::uint32_t BaseAlign>
void serialise(Serialiser<BaseAlign>& ss, const std::vector<Sphere>& arr) {
  std::uint32_t size = arr.size(); // size_t is not portable
  ss << size;
  for (const auto& s : arr) {
    ss << s;
  }
}

template <std::uint32_t BaseAlign>
void serialise(Serialiser<BaseAlign>& ss, const std::vector<Disc>& arr) {
  std::uint32_t size = arr.size(); // size_t is not portable
  ss << size;
  for (const auto& d : arr) {
    ss << d;
  }
}
