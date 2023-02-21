// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains a simple wrapper around an Embree device/scene and associated utility functions.

#pragma once

#include "geometry.hpp"
#include "shapes.hpp"

namespace {

RTCRayHit convertHitRecord(const embree_utils::HitRecord& hit) {
  RTCRayHit rh;
  rh.ray.org_x = hit.r.origin.x;
  rh.ray.org_y = hit.r.origin.y;
  rh.ray.org_z = hit.r.origin.z;
  rh.ray.dir_x = hit.r.direction.x;
  rh.ray.dir_y = hit.r.direction.y;
  rh.ray.dir_z = hit.r.direction.z;
  rh.ray.tnear  = 0.f;
  rh.ray.tfar   = hit.r.tMax;
  rh.hit.Ng_x = hit.normal.x;
  rh.hit.Ng_y = hit.normal.y;
  rh.hit.Ng_z = hit.normal.z;
  rh.hit.primID = hit.primID;
  rh.hit.geomID = hit.geomID;
  rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
  rh.ray.mask = 0xFFFFFFFF;
  rh.ray.flags = 0;
  rh.ray.time = 0.f;
  return rh;
}

embree_utils::HitRecord convertHitRecord(const RTCRayHit& rh) {
  embree_utils::HitRecord hit(
    embree_utils::Vec3fa(rh.ray.org_x, rh.ray.org_y, rh.ray.org_z),
    embree_utils::Vec3fa(rh.ray.dir_x, rh.ray.dir_y, rh.ray.dir_z)
  );
  hit.r.tMin = rh.ray.tnear;
  hit.r.tMax = rh.ray.tfar;
  hit.normal = embree_utils::Vec3fa(rh.hit.Ng_x, rh.hit.Ng_y, rh.hit.Ng_z).normalized();
  hit.geomID = rh.hit.geomID;
  hit.primID = rh.hit.primID;
  return hit;
}

} // end anon namespace

namespace embree_utils {

class EmbreeScene {
  RTCDevice device;
  RTCScene scene   = rtcNewScene(device);

public:
  EmbreeScene() : device(rtcNewDevice(nullptr)), scene(rtcNewScene(device)) {
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_COMPACT | RTC_SCENE_FLAG_ROBUST);
    auto err = rtcGetDeviceError(device);
    if (err != RTC_ERROR_NONE) {
      throw std::runtime_error("Could not acquire an RTCDevice.");
    }
  }

  virtual ~EmbreeScene() {
    rtcReleaseDevice(device);
  }

  RTCDevice& getDevice() {
    return device;
  }

  RTCScene& getScene() {
    return scene;
  }

  void commitScene() {
    rtcCommitScene(scene);
  }

  void intersect(std::vector<RTCRayHit>& hitStream, std::size_t numParallelJobs) {
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    // Check we can intersect rays in parallel batches:
    if (hitStream.size() % numParallelJobs) {
      throw std::logic_error("Can not split ray stream into the specified job size.");
    }
    auto raysPerJob = hitStream.size() / numParallelJobs;
    auto jobs = hitStream.size() / raysPerJob;

    #pragma omp parallel for schedule(auto)
    for (auto j = 0u; j < jobs; ++j) {
      auto startIndex = j * raysPerJob;
      rtcIntersect1M(scene, &context, &hitStream[startIndex], raysPerJob, sizeof(RTCRayHit));
    }
  }

  void occluded(std::vector<RTCRayHit>& hitStream, std::size_t numParallelJobs) {
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    // Test if we can intersect rays in parallel:
    if (hitStream.size() % numParallelJobs) {
      throw std::logic_error("Can not split ray stream into the specified job size.");
    }
    auto raysPerJob = hitStream.size() / numParallelJobs;
    auto jobs = hitStream.size() / raysPerJob;

    #pragma omp parallel for schedule(auto)
    for (auto j = 0u; j < jobs; ++j) {
      auto startIndex = j * raysPerJob;
      rtcOccluded1M(scene, &context, &(hitStream[startIndex].ray), raysPerJob, sizeof(RTCRayHit));
    }
  }

  void addSphere(const embree_utils::Vec3fa& pos, float radius) {
    embree_utils::addSphere(device, scene, pos, radius);
  }

  void addTriMesh(ConstArrayRef<embree_utils::Vec3fa> vertices, ConstArrayRef<std::uint16_t> triIndices) {
    embree_utils::addTriMesh(device, scene, vertices, triIndices);
  }

  void addDisc(const embree_utils::Vec3fa& normal, const embree_utils::Vec3fa& pos, float radius) {
    embree_utils::addDisc(device, scene, normal, pos, radius);
  }
};


} // end namespace embree_utils
