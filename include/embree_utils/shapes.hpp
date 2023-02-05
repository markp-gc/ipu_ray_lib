// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains utility functions for creating and attaching basic primitives to an Embree scene.

#pragma once

#include "geometry.hpp"
#include "Arrays.hpp"

#include <iostream>


namespace {

struct Tri { std::uint32_t v0, v1, v2; };

} // end anon namespace

namespace embree_utils {

unsigned int addTriMesh(RTCDevice device, RTCScene scene,
                        ConstArrayRef<embree_utils::Vec3fa> vertices,
                        ConstArrayRef<std::uint16_t> triIndices) {
  // create a triangulated cube with 12 triangles and 8 vertices
  RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

  embree_utils::Vec3fa* verts = (embree_utils::Vec3fa*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(embree_utils::Vec3fa), vertices.size());
  auto vertIndex = 0u;
  for (auto& v : vertices) {
    verts[vertIndex] = v;
    vertIndex += 1;
  }

  auto numTris = triIndices.size() / 3;
  if (triIndices.size() % 3) {
    throw std::logic_error("Number of triangle indices not divisible by 3.");
  }
  Tri* tris = (Tri*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Tri), numTris);
  auto triIndex = 0u;
  for (auto t = 0u; t < numTris; ++t) {
    tris[t].v0 = triIndices[triIndex]; ++triIndex;
    tris[t].v1 = triIndices[triIndex]; ++triIndex;
    tris[t].v2 = triIndices[triIndex]; ++triIndex;
  }

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(scene, mesh);
  rtcReleaseGeometry(mesh);
  return geomID;
}

void addSphere(RTCDevice device, RTCScene scene, const embree_utils::Vec3fa& pos, float radius) {
  RTCGeometry sphere = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
  float* loc = (float*)rtcSetNewGeometryBuffer(sphere, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, 4 * sizeof(float), 1);
  loc[0] = pos.x;
  loc[1] = pos.y;
  loc[2] = pos.z;
  loc[3] = radius;

  rtcCommitGeometry(sphere);
  rtcAttachGeometry(scene, sphere);
  rtcReleaseGeometry(sphere);
}

unsigned int addDisc(RTCDevice device, RTCScene scene, const embree_utils::Vec3fa& pos, const embree_utils::Vec3fa& normal, float radius) {
  RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
  float* loc = (float*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, 4 * sizeof(float), 1);
  loc[0] = pos.x;
  loc[1] = pos.y;
  loc[2] = pos.z;
  loc[3] = radius;

  embree_utils::Vec3fa* normals = (embree_utils::Vec3fa*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_NORMAL, 0, RTC_FORMAT_FLOAT3, sizeof(embree_utils::Vec3fa), 1);
  normals[0] = normal;

  rtcCommitGeometry(geom);
  auto geomID = rtcAttachGeometry(scene, geom);
  rtcReleaseGeometry(geom);
  return geomID;
}

} // end namespace embree_utils
