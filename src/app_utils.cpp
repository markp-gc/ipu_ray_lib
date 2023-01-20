// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <app_utils.hpp>

#include <Primitives.hpp>
#include <Scene.hpp>
#include <Mesh.hpp>
#include <xoshiro.hpp>

#include <regex>
#include <optional>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <functional>
#include <vector>
#include <random>

void initPerspectiveRayStream(std::vector<embree_utils::TraceResult>& rayStream,
                              const cv::Mat& image,
                              const CropWindow& window,
                              float fovRadians,
                              xoshiro::Generator* gen) {
  const auto rayOrigin = embree_utils::Vec3fa(0, 0, 0);

  // Do trig outside of loop:
  float s, c;
  sincos(fovRadians / 2.f, s, c);
  const auto fovTanTheta = s / c;

  std::normal_distribution<float> d{0.f, .25f};

  auto i = 0u;
  for (std::uint32_t r = window.r; r < window.r + window.h; ++r) {
    for (std::uint32_t c = window.c; c < window.c + window.w; ++c) {
      float pu = r;
      float pv = c;
      if (gen) {
        pu += d(*gen);
        pv += d(*gen);
      }
      const auto rayDir = pixelToRayDir(pv, pu, image.cols, image.rows, fovTanTheta);
      rayStream[i].h = embree_utils::HitRecord(rayOrigin, rayDir);
      rayStream[i].p = embree_utils::PixelCoord{r, c};
      i += 1;
    }
  }
}

void zeroRgb(std::vector<embree_utils::TraceResult>& rayStream) {
  for (auto& r : rayStream) {
    r.rgb = embree_utils::Vec3fa(0, 0, 0);
  }
}

void scaleRgb(std::vector<embree_utils::TraceResult>& rayStream, float scale) {
  for (auto& r : rayStream) {
    r.rgb *= scale;
  }
}

unsigned visualiseHits(const std::vector<embree_utils::TraceResult>& rayStream, const SceneRef& data, cv::Mat& image, VisualiseMode mode) {
  // Note: Remember OpenCV is BGR by default when modifying these functions:
  auto rgbFunc = [&](const embree_utils::TraceResult& tr) {
    return cv::Vec3f(tr.rgb.z, tr.rgb.y, tr.rgb.x);
  };
  auto primFunc = [&](const embree_utils::TraceResult& tr) {
    const auto& hit = tr.h;
    if (hit.geomID == embree_utils::HitRecord::InvalidGeomID) {
      return cv::Vec3f(0.f, 0.f, 0.f);
    }
    // Zero indicates no hit so increment all ids by one:
    auto primId = hit.primID + 1;
    auto geomId = hit.geomID + 1;
    auto matId = data.matIDs[hit.geomID] + 1;
    return cv::Vec3f(geomId, primId, matId);
  };
  auto normalFunc = [&](const embree_utils::TraceResult& tr) {
    const auto& hit = tr.h;
    if (hit.geomID == embree_utils::HitRecord::InvalidGeomID) {
      return cv::Vec3f(0.f, 0.f, 0.f);
    }
    auto n = hit.normal;
    return cv::Vec3f(n.z, n.y, n.x);
  };
  auto tfarFunc = [&](const embree_utils::TraceResult& tr) {
    const auto& hit = tr.h;
    return cv::Vec3f(hit.r.tMax, hit.r.tMax, hit.r.tMax);
  };
  auto colFunc = [&](const embree_utils::TraceResult& tr) {
    const auto& hit = tr.h;
    if (hit.geomID == embree_utils::HitRecord::InvalidGeomID) {
      return cv::Vec3f(0.f, 0.f, 0.f);
    }
    auto id = data.matIDs[hit.geomID];
    auto c = data.materials[id].albedo;
    return cv::Vec3f(c.z, c.y, c.x);
  };
  auto hpFunc = [&](const embree_utils::TraceResult& tr) {
    const auto& hit = tr.h;
    return cv::Vec3f(hit.r.origin.z, hit.r.origin.y, hit.r.origin.x);
  };

  std::function<cv::Vec3f(const embree_utils::TraceResult&)> visFunc[] = {
    rgbFunc, primFunc, normalFunc, tfarFunc, colFunc, hpFunc
  };

  // Process results in parallel:
  std::atomic<unsigned> hitCount(0);
  #pragma omp parallel for schedule(auto)
  for (auto rsItr = rayStream.begin(); rsItr != rayStream.end(); ++rsItr) {
    auto& hit = rsItr->h;
    const auto& pixCoord = rsItr->p;
    image.at<cv::Vec3f>(pixCoord.u, pixCoord.v) = visFunc[mode](*rsItr);
    if (hit.geomID != hit.InvalidGeomID) {
      hitCount += 1;
    }
  }

  return hitCount;
}

const Primitive* getPrimitive(GeomRef geom, const SceneDescription& scene) {
  switch (geom.type) {
    case GeomType::Mesh:
      return &scene.meshes[geom.index];
    break;
    case GeomType::Sphere:
      return &scene.spheres[geom.index];
    break;
    case GeomType::Disc:
      return &scene.discs[geom.index];
    break;
  }

  throw std::logic_error("Invalid GeomRef.");
}

std::vector<RTCBuildPrimitive> makeBuildPrimitivesForEmbree(const SceneData& data, const SceneDescription& scene) {
  // Make duplicates of primitives for Embree custom BVH build.
  // Embree only needs to know the bounds and IDs:
  embree_utils::Bounds3d extent;
  std::vector<RTCBuildPrimitive> buildPrimitives;
  buildPrimitives.reserve(data.geometry.size());
  bool separateTriangles = true;
  for (auto i = 0u; i < data.geometry.size(); ++i) {
    RTCBuildPrimitive bp;
    bp.geomID = i; // Note: Geometry creation ordering must match this indexing.

    auto* p = getPrimitive(data.geometry[i], scene);
    const auto bbox = p->getBoundingBox();
    extent += bbox;

    const HostTriangleMesh* tri = nullptr;
    if (separateTriangles && (tri = dynamic_cast<const HostTriangleMesh*>(p))) {
      // Special handling for triangle meshes.
      // Individual trianglea are recorded as separate primitives:
      for (auto pid = 0u; pid < tri->triangles.size(); ++pid) {
        const auto triBounds = tri->getTriangleBoundingBox(pid);
        bp.lower_x = triBounds.min.x;
        bp.lower_y = triBounds.min.y;
        bp.lower_z = triBounds.min.z;
        bp.upper_x = triBounds.max.x;
        bp.upper_y = triBounds.max.y;
        bp.upper_z = triBounds.max.z;
        bp.primID = pid;
        buildPrimitives.push_back(bp);
      }
    } else {
      bp.lower_x = bbox.min.x;
      bp.lower_y = bbox.min.y;
      bp.lower_z = bbox.min.z;
      bp.upper_x = bbox.max.x;
      bp.upper_y = bbox.max.y;
      bp.upper_z = bbox.max.z;
      bp.primID = 0;
      buildPrimitives.push_back(bp);
    }
  }
  
  return buildPrimitives;
}

void setupLogging(const boost::program_options::variables_map& args) {
  std::map<std::string, spdlog::level::level_enum> levelFromStr = {
    {"trace", spdlog::level::trace},
    {"debug", spdlog::level::debug},
    {"info", spdlog::level::info},
    {"warn", spdlog::level::warn},
    {"err", spdlog::level::err},
    {"critical", spdlog::level::critical},
    {"off", spdlog::level::off}
  };

  const auto levelStr = args["log-level"].as<std::string>();
  try {
    spdlog::set_level(levelFromStr.at(levelStr));
  } catch (const std::exception& e) {
    std::stringstream ss;
    ss << "Invalid log-level: '" << levelStr << "'";
    throw std::runtime_error(ss.str());
  }
  spdlog::set_pattern("[%H:%M:%S.%f] [%L] [%t] %v");
}

std::optional<CropWindow> parseCropString(const std::string& cropFmt) {
  if (cropFmt.empty()) {
    return {};
  } else {
    // Parse the format string:
    ipu_utils::logger()->debug("Crop format string: {}", cropFmt);
    std::smatch matches;
    std::regex_search(cropFmt, matches, std::regex("(\\d+)x(\\d+)\\+(\\d+)\\+(\\d+)"));
    if (matches.size() != 5) {
      ipu_utils::logger()->error("Could not parse --crop argument: '{}'", cropFmt);
      throw std::runtime_error("Badly formatted string used for --crop.");
    }
    return std::optional<CropWindow>(
      CropWindow{
        std::atoi(matches.str(1).c_str()),
        std::atoi(matches.str(2).c_str()),
        std::atoi(matches.str(3).c_str()),
        std::atoi(matches.str(4).c_str())
      }
    );
  }
}
