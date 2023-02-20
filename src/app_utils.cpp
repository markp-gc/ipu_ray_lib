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
                              const SceneRef& data,
                              xoshiro::Generator* gen) {
  const auto rayOrigin = embree_utils::Vec3fa(0, 0, 0);

  // Do trig outside of loop:
  float s, c;
  sincos(data.fovRadians / 2.f, s, c);
  const auto fovTanTheta = s / c;

  std::normal_distribution<float> d{0.f, data.antiAliasScale};

  auto i = 0u;
  for (std::uint32_t r = data.window.r; r < data.window.r + data.window.h; ++r) {
    for (std::uint32_t c = data.window.c; c < data.window.c + data.window.w; ++c) {
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

  ipu_utils::logger()->trace("Visualise start");

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

  ipu_utils::logger()->trace("Visualise end");

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

std::unique_ptr<PathTraceSettings> makePathTraceSettings(const boost::program_options::variables_map& args) {
  auto settings = std::make_unique<PathTraceSettings>();
  auto rngSeed = args["seed"].as<std::uint64_t>();
  settings->sampler = xoshiro::Generator(rngSeed);
  settings->maxPathLength = args["max-path-length"].as<std::uint32_t>();
  settings->roulletteStartDepth = args["roulette-start-depth"].as<std::uint32_t>();
  settings->samplesPerPixel = args["samples"].as<std::uint32_t>();
  if (args.at("visualise").as<std::string>() != "rgb") {
    throw std::runtime_error("Running path-tracing without visualise=rgb is not advised.");
  }
  return settings;
}

// Load or build the scene description (depending on args).
// This is a high level description (i.e. in case of loading
// from file the scne is imported into data structures that
// mirror the file import format).
SceneDescription buildSceneDescription(const boost::program_options::variables_map& args) {
  // Create the high level scene description:
  auto meshFile = args["mesh-file"].as<std::string>();

  SceneDescription scene;
  if (meshFile.empty()) {
    // If no meshfile is provided then default to the box. In this case a
    // hard coded mesh file is displayed using one of the boxes as a plinth:
    meshFile = "../assets/monkey_bust.glb";
    scene = makeCornellBoxScene(meshFile, args["box-only"].as<bool>());
  } else {
    // Otherwise load only the specified scene:
    scene = importScene(meshFile, args["load-normals"].as<bool>());
  }

  if (args["render-mode"].as<std::string>() == "path-trace") {
    scene.pathTrace = makePathTraceSettings(args);
  }

  return scene;
}

// Build efficient scene representations for both Embree and our custom CPU/IPU renderers/
//
// The scene description needs to be converted into a compact representation
// that can be shared (as far as possible) between Embree, CPU, and IPU
// renders.
//
// Note: creation order is important in all cases because geomIDs
// (Embree concept) are used to retrieve primitives during BVH traversal)
// Mapping between materials and primitives also depends on a consistent order.
std::pair<SceneData, embree_utils::EmbreeScene> buildSceneData(const SceneDescription& scene) {
  SceneData data;

  // We need a compact representation for multiple meshes that we can transfer
  // to the device easily. Append all the triangle buffer indices, vertices,
  // and normals into unified arrays:
  data.meshInfo.reserve(scene.meshes.size());
  for (const auto& m : scene.meshes) {
    data.meshInfo.emplace_back(
      MeshInfo{
        (std::uint32_t)data.meshTris.size(), (std::uint32_t)data.meshVerts.size(),
        (std::uint32_t)m.triangles.size(), (std::uint32_t)m.vertices.size()
      }
    );

    for (const auto& t : m.triangles) {
      data.meshTris.push_back(t);
    }
    for (const auto& v : m.vertices) {
      data.meshVerts.push_back(v);
    }
    for (const auto& n : m.normals) {
      data.meshNormals.push_back(n);
    }
  }

  // Initialise Embree:
  embree_utils::EmbreeScene embreeScene;

  // Create both the Embree and custom representations of all primitives:
  for (auto i = 0u; i < scene.meshes.size(); ++i) {
    auto& m = scene.meshes[i];
    data.geometry.emplace_back(i, GeomType::Mesh);
    embreeScene.addTriMesh(
      m.vertices,
      ConstArrayRef<std::uint16_t>::reinterpret(m.triangles.data(), m.triangles.size()));
  }

  for (auto i = 0u; i < scene.spheres.size(); ++i) {
    auto& s = scene.spheres[i];
    data.geometry.emplace_back(i, GeomType::Sphere);
    embreeScene.addSphere(embree_utils::Vec3fa(s.x, s.y, s.z), s.radius);
  }

  for (auto i = 0u; i < scene.discs.size(); ++i) {
    auto& d = scene.discs[i];
    data.geometry.emplace_back(i, GeomType::Disc);
    embreeScene.addDisc(embree_utils::Vec3fa(d.nx, d.ny, d.nz), embree_utils::Vec3fa(d.cx, d.cy, d.cz), d.r);
  }

  data.materials = scene.materials;
  data.matIDs = scene.matIDs;

  auto bvhStartTime = std::chrono::steady_clock::now();

  auto buildPrimitives = makeBuildPrimitivesForEmbree(data, scene);

  // Build our own BVH (still using Embree to build it):
  embree_utils::BvhBuilder builder(embreeScene.getDevice());
  builder.build(buildPrimitives);
  std::uint32_t maxDepth;

  // Convert our custom BVH into a compact form (i.e. convert the tree to a linear array):
  data.bvhNodes = buildCompactBvh(builder.getRoot(), builder.nodeCount(), data.bvhMaxDepth);
  buildPrimitives.clear(); // Embree is done with this now so free the space

  auto bvhEndTime = std::chrono::steady_clock::now();
  auto bvhSecs = std::chrono::duration<double>(bvhEndTime - bvhStartTime).count();

  ipu_utils::logger()->info("Compact BVH build time: {} seconds", bvhSecs);
  ipu_utils::logger()->debug("Max leaf depth in BVH: {}", data.bvhMaxDepth);

  return {data, embreeScene};
}
