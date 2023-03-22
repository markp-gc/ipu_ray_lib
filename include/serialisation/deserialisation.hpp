// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

// Functions for deserialising commonly used types:

#include <serialisation/Deserialiser.hpp>
#include <CompactBVH2Node.hpp>
#include <Arrays.hpp>
#include <Scene.hpp>

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
  std::uint32_t size;
  d >> size;
  arr.clear();
  arr.resize(size);
  d.read(arr.data(), size);
}
#endif

template <typename T, std::uint32_t BaseAlign>
ArrayRef<T> deserialiseArrayRef(Deserialiser<BaseAlign>& d) {
  std::uint32_t size;
  d >> size;
  d.template skipPadding<T>();
  ArrayRef<T> arr((T*)d.getPtr(), size);
  d.skip(size * sizeof(T));
  return arr;
}

template <std::uint32_t BaseAlign>
void deserialise(Deserialiser<BaseAlign>& d, SceneRef& s) {
  s.geometry = deserialiseArrayRef<GeomRef>(d);
  s.meshInfo = deserialiseArrayRef<MeshInfo>(d);
  s.meshTris = deserialiseArrayRef<Triangle>(d);
  s.meshVerts = deserialiseArrayRef<embree_utils::Vec3fa>(d);
  s.meshNormals = deserialiseArrayRef<embree_utils::Vec3fa>(d);
  s.matIDs = deserialiseArrayRef<std::uint32_t>(d);
  s.materials = deserialiseArrayRef<Material>(d);
  s.bvhNodes = deserialiseArrayRef<CompactBVH2Node>(d);
  d >> s.maxLeafDepth;
  d >> s.imageWidth;
  d >> s.imageHeight;
  d >> s.fovRadians;
  d >> s.antiAliasScale;
  d >> s.maxPathLength;
  d >> s.rouletteStartDepth;
  d >> s.samplesPerPixel;
}
