#include <embree_utils/bvh.hpp>
#include <embree_utils/shapes.hpp>
#include <embree_utils/EmbreeScene.hpp>
#include <IpuScene.hpp>
#include <scene_utils.hpp>
#include <CompactBvhBuild.hpp>
#include <CompactBvh.hpp>
#include <Primitives.hpp>
#include <Scene.hpp>
#include <Mesh.hpp>
#include <xoshiro.hpp>
#include <Render.hpp>
#include <BxDF.hpp>

#include <boost/program_options.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <functional>
#include <vector>
#include <map>
#include <random>

void initPerspectiveRayStream(std::vector<embree_utils::TraceResult>& rayStream,
                              const cv::Mat& image,
                              xoshiro::Generator* gen = nullptr) {
  constexpr float fov = embree_utils::Piby4;
  const auto rayOrigin = embree_utils::Vec3fa(0, 0, 0);

  std::normal_distribution<float> d{0.f, .25f};

  auto i = 0u;
  for (std::uint32_t r = 0u; r < image.rows; ++r) {
    for (std::uint32_t c = 0u; c < image.cols; ++c) {
      float pu = r;
      float pv = c;
      if (gen) {
        pu += d(*gen);
        pv += d(*gen);
      }
      const auto rayDir = pixelToRayDir(pv, pu, image.cols, image.rows, fov);
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

enum VisualiseMode {
  RGB,
  GEOM_AND_PRIM_ID,
  NORMAL,
  RAY_TFAR,
  MAT_COLOR,
  HIT_POINT
};

enum RenderMode {
  SHADOW_TRACE,
  PATH_TRACE
};

std::map<std::string, VisualiseMode> visStrMap = {
  {"rgb", RGB},
  {"normal", NORMAL},
  {"hitpoint", HIT_POINT},
  {"tfar", RAY_TFAR},
  {"color", MAT_COLOR},
  {"id", GEOM_AND_PRIM_ID}
};

std::map<std::string, RenderMode> renderStrMap = {
  {"shadow-trace", SHADOW_TRACE},
  {"path-trace", PATH_TRACE}
};

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

std::vector<embree_utils::TraceResult> renderEmbree(const SceneRef& data, embree_utils::EmbreeScene& embreeScene, cv::Mat& image) {
  std::vector<embree_utils::TraceResult> rayStream(image.rows * image.cols);
  initPerspectiveRayStream(rayStream, image);
  zeroRgb(rayStream);

  embreeScene.commitScene();

  // Convert stream:
  std::vector<RTCRayHit> hitStream;
  hitStream.reserve(rayStream.size());
  for (const auto& r : rayStream) {
    hitStream.push_back(convertHitRecord(r.h));
  }

  if (((std::size_t)(&hitStream[0].ray)) % 16) {
    throw std::logic_error("First element of ray stream is not 16-byte aligned.");
  }
  auto startTime = std::chrono::steady_clock::now();
  embreeScene.intersect(hitStream, image.rows);

  // For Embree we have to save hit records before shadow ray calcs
  // (because we want to use the streaming version of occlusion test
  // with the same ray stream for performance reasons):
  #pragma omp parallel for schedule(auto)
  for (auto i = 0u; i < hitStream.size(); ++i) {
    auto& rh = hitStream[i];
    rayStream[i].h = convertHitRecord(rh);
  }

  // Setup a shadow test:
  embree_utils::Vec3fa light(18, 257, -1060);
  #pragma omp parallel for schedule(auto)
  for (auto r = 0u; r < hitStream.size(); ++r) {
    auto& rh = hitStream[r];
    // // advance origin to hit point
    if (std::isfinite(rh.ray.tfar)) {
      rh.ray.org_x += rh.ray.dir_x * rh.ray.tfar;
      rh.ray.org_y += rh.ray.dir_y * rh.ray.tfar;
      rh.ray.org_z += rh.ray.dir_z * rh.ray.tfar;
      // Setup ray for an occlusion query:
      // point ray at light:
      float dx = light.x - rh.ray.org_x;
      float dy = light.y - rh.ray.org_y;
      float dz = light.z - rh.ray.org_z;
      float d = std::sqrt(dx*dx + dy*dy + dz*dz);
      float norm = 1.f / d;
      rh.ray.dir_x = dx * norm;
      rh.ray.dir_y = dy * norm;
      rh.ray.dir_z = dz * norm;
      rh.ray.tnear = data.shadowRayOffset;
      rh.ray.tfar = d;
    } else {
      rh.ray.tfar = -std::numeric_limits<float>::infinity();
    }
  }
  embreeScene.occluded(hitStream, image.rows);
  auto endTime = std::chrono::steady_clock::now();

  // Store shadowing/shading result:
  #pragma omp parallel for schedule(auto)
  for (auto i = 0u; i < hitStream.size(); ++i) {
    auto& rh = hitStream[i];
    if (rh.hit.primID == RTC_INVALID_GEOMETRY_ID) {
      continue;
    }
    auto matRgb = data.materials[data.matIDs[rh.hit.geomID]].albedo;
    auto color = matRgb * 0.05f; // ambient
    if (rh.ray.tfar != -std::numeric_limits<float>::infinity()) {
      float norm = 1.f / std::sqrt(rh.hit.Ng_x*rh.hit.Ng_x + rh.hit.Ng_y*rh.hit.Ng_y + rh.hit.Ng_z*rh.hit.Ng_z);
      auto costh =
        norm * (rh.hit.Ng_x * rh.ray.dir_x) +
        norm * (rh.hit.Ng_y * rh.ray.dir_y) +
        norm * (rh.hit.Ng_z * rh.ray.dir_z);
      color += matRgb * costh;
    }
    rayStream[i].rgb = color;
  }

  auto secs = std::chrono::duration<double>(endTime - startTime).count();
  auto rayRate = rayStream.size() / secs;
  ipu_utils::logger()->info("Embree ray rate: {} rays/sec", rayRate);

  return rayStream;
}

template <class T>
void pathTrace(const SceneRef& sceneRef,
               SceneDescription& scene,
               const CompactBvh& bvh,
               embree_utils::TraceResult& result,
               T& primLookupFunc) {
  using namespace embree_utils;
  auto& hit = result.h;
  Vec3fa throughput(1.f, 1.f, 1.f);
  Vec3fa color(0.f, 0.f, 0.f);

  for (auto i = 0u; i < scene.pathTrace->maxPathLength; ++i) {
    // Reset ray limits for next bounce:
    hit.r.tMin = sceneRef.shadowRayOffset; // offset rays to avoid self intersection.
    hit.r.tMax = std::numeric_limits<float>::infinity();
    auto intersected = bvh.intersect(hit.r, primLookupFunc);

    if (intersected) {
      updateHit(intersected, hit);
      const auto& material = sceneRef.materials[sceneRef.matIDs[hit.geomID]];

      if (material.emissive) {
        color += throughput * material.emission;
      }

      if (material.type == Material::Type::Diffuse) {
        // CPU Generator not thread safe:
        float u1, u2;
        #pragma omp critical(sample)
        {
          u1 = scene.pathTrace->sampler.uniform_0_1();
          u2 = scene.pathTrace->sampler.uniform_0_1();
        }
        hit.r.direction = sampleDiffuse(hit.normal, u1, u2);
        // Update throughput
        //const float w = std::abs(wiWorld.dot(normal));
        //const float pdf = cosineHemispherePdf(wiTangent);
        // The terms w / (Pi * pdf) all cancel for diffuse throughput:
        throughput *= material.albedo; // * (w / (Pi * pdf)); // PDF terms cancel for cosine weighted samples
        //throughput *= material.albedo * (wiTangent.z * 2.f); // Apply PDF for hemisphere samples (sampleDir is in tangent space so cos(theta) == z-coord).
      } else if (material.type == Material::Type::Specular) {
        hit.r.direction = reflect(hit.r.direction, hit.normal);
        throughput *= material.albedo;
      } else if (material.type == Material::Type::Refractive) {
        float u1;
        #pragma omp critical(sample)
        { u1 = scene.pathTrace->sampler.uniform_0_1(); }
        constexpr float refractiveIndex = 1.52f;
        hit.r.direction = refract(hit.r, hit.normal, refractiveIndex, u1);
        throughput *= material.albedo * refractiveIndex;
      } else {
        // Mark an error:
        result.rgb *= std::numeric_limits<float>::quiet_NaN();
      }
    } else {
      break;
    }

    // Random stopping:
    if (i > scene.pathTrace->roulletteStartDepth) {
      float u1;
      #pragma omp critical(sample)
      { u1 = scene.pathTrace->sampler.uniform_0_1(); }
      if (evaluateRoulette(u1, throughput)) { break; }
    }
  }

  result.rgb += color;
}

std::vector<embree_utils::TraceResult> renderCPU(
  SceneRef& sceneRef, cv::Mat& image, SceneDescription& scene
) {
  // Use the compiled variant of a mesh (to match IPU implementation
  // as closely as possible). A compiled mesh's internal storage
  // is setup to reference the correct parts of the unified arrays
  // in the scene description:
  std::vector<CompiledTriangleMesh> meshes;
  meshes.reserve(sceneRef.meshInfo.size());

  #pragma omp parallel for schedule(auto)
  for (auto s = 0u; s < sceneRef.meshInfo.size(); ++s) {
    const auto& info = sceneRef.meshInfo[s];
    meshes.emplace_back(
      embree_utils::Bounds3d(), // Don't actually need this bound box for rendering...
      ConstArrayRef(&sceneRef.meshTris[info.firstIndex], info.numTriangles),
      ConstArrayRef(&sceneRef.meshVerts[info.firstVertex], info.numVertices)
    );
  }

  std::vector<embree_utils::TraceResult> rayStream(image.rows * image.cols);
  initPerspectiveRayStream(rayStream, image);
  zeroRgb(rayStream);

  // Time just the intersections with the compact BVH:
  auto startTime = std::chrono::steady_clock::now();

  // Make a CompactBvh object for our custom CPU ray-tracer.
  // A CompactBvh wraps the scene ref BVH nodes:
  CompactBvh bvh(sceneRef.bvhNodes, sceneRef.maxLeafDepth);

  auto primLookup = [&](std::uint16_t geomID, std::uint32_t primID) {
    const auto& geom = sceneRef.geometry[geomID];
    return getPrimitive(geom, scene);
  };

  if (scene.pathTrace) {
    for (auto s = 0u; s < scene.pathTrace->samplesPerPixel; ++s) {
      // Regenerate new camera rays at each sample step:
      initPerspectiveRayStream(rayStream, image, &scene.pathTrace->sampler);
      #pragma omp parallel for schedule(auto)
      for (auto itr = rayStream.begin(); itr != rayStream.end(); ++itr) {
        pathTrace(sceneRef, scene, bvh, *itr, primLookup);
      }
    }
    scaleRgb(rayStream, 1.f / scene.pathTrace->samplesPerPixel);
  } else {
    embree_utils::Vec3fa lightPos(18, 257, -1060); // hard coded light position for testing
    #pragma omp parallel for schedule(auto)
    for (auto itr = rayStream.begin(); itr != rayStream.end(); ++itr) {
      traceShadowRay(
        bvh,
        sceneRef.matIDs, sceneRef.materials, sceneRef.shadowRayOffset,
        .05f, // ambient light factor
        *itr, primLookup, lightPos);
    }
  }

  auto endTime = std::chrono::steady_clock::now();
  auto secs = std::chrono::duration<double>(endTime - startTime).count();
  auto castsPerRay = scene.pathTrace ? scene.pathTrace->samplesPerPixel * scene.pathTrace->maxPathLength : 1;
  auto rateString = scene.pathTrace ? "paths" : "rays";
  auto rate = rayStream.size() * castsPerRay / secs;
  ipu_utils::logger()->info("CPU time: {}", secs);
  ipu_utils::logger()->info("CPU {} per second: {} ", rateString, rate);

  return rayStream;
}

std::vector<embree_utils::TraceResult> renderIPU(
  SceneRef& sceneRef, cv::Mat& image,
  const std::vector<Sphere>& spheres,
  const std::vector<Disc>& discs,
  const boost::program_options::variables_map& args)
{
  std::vector<embree_utils::TraceResult> rayStream(image.rows * image.cols);
  initPerspectiveRayStream(rayStream, image);
  zeroRgb(rayStream);

  auto ipus = args["ipus"].as<std::uint32_t>();
  IpuScene ipuScene(spheres, discs, sceneRef, rayStream);
  ipuScene.setRuntimeConfig(
    ipu_utils::RuntimeConfig {
      ipus, // numIpus;
      ipus, // numReplicas;
      "ipu_ray_trace", // exeName;
      false, // useIpuModel;
      false, // saveExe;
      false, // loadExe;
      false, // compileOnly;
      true   // deferredAttach;
    }
  );

  // For now we set a large timeout because the current implementation
  // never returns to the host during rendering:
  ipu_utils::GraphManager().run(ipuScene,
                                {{"target.hostSyncTimeout", "10000"}});

  if (sceneRef.pathTrace) {
    scaleRgb(rayStream, 1.f / sceneRef.samplesPerPixel);
  }

  // NOTE: Currently only one tile used on each replica
  // so single IPU benchmark should be 1440x faster with
  // perfect replication across tiles (we know scaling is
  // perfect in that scenario):
  auto secs = ipuScene.getTraceTimeSecs();
  auto castsPerRay = sceneRef.pathTrace ? sceneRef.samplesPerPixel : 1;
  auto rateString = sceneRef.pathTrace ? "paths" : "rays";
  auto rate = rayStream.size() * castsPerRay / secs;
  ipu_utils::logger()->info("IPU time: {}", secs);
  ipu_utils::logger()->info("IPU {} per second: {} ", rateString, rate);

  return rayStream;
}

void addOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("help", "Show command help.")
  ("ipus", po::value<std::uint32_t>()->default_value(4), "Select number of IPUs (each IPU will be a replica).")
  ("width,w", po::value<std::int32_t>()->default_value(768), "Set rendered image width.")
  ("height,h", po::value<std::int32_t>()->default_value(432), "Set rendered image height.")
  ("mesh-file", po::value<std::string>()->default_value(std::string()), "Mesh file to include in render. Format must be supported by libassimp.")
  ("visualise", po::value<std::string>()->default_value("rgb"), "Choose the render output values to test/visualise. One of [rgb, normal, hitpoint, tfar, color, id]")
  ("ray-epsilon", po::value<float>()->default_value(0.0035f), "Shadow ray min intersect distance (applied for Embree, CPU, and IPU).")
  ("render-mode", po::value<std::string>()->default_value("path-trace"), "Choose type of render from [shadow-trace, path-trace]. To see result set visualise=rgb")
  ("max-path-length", po::value<std::uint32_t>()->default_value(12), "Max path length for path tracing.")
  ("roulette-start-depth", po::value<std::uint32_t>()->default_value(5), "Path length after which rays can be randomly terminated with prob. inversely proportional to their throughput.")
  ("samples", po::value<std::uint32_t>()->default_value(256), "Number of samples per pixel for path tracing.")
  ("seed", po::value<std::uint64_t>()->default_value(1442), "RNG seed.")
  ("box-only", po::bool_switch()->default_value(false), "Only render the original Cornell box scene without extra elements.")
  ("ipu-only", po::bool_switch()->default_value(false), "Only render on IPU (e.g. if you don't want to wait for slow CPU path tracing).");
}

boost::program_options::variables_map parseOptions(int argc, char** argv, boost::program_options::options_description& desc) {
  namespace po = boost::program_options;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    throw std::runtime_error("Show help");
  }

  try {
    const auto str = vm.at("visualise").as<std::string>();
    const auto v = visStrMap.at(str);
  } catch (const std::exception& e) {
    throw po::validation_error(po::validation_error::invalid_option_value, "visualise");
  }

  try {
    const auto str = vm.at("render-mode").as<std::string>();
    const auto v = renderStrMap.at(str);
  } catch (const std::exception& e) {
    throw po::validation_error(po::validation_error::invalid_option_value, "render-mode");
  }

  po::notify(vm);
  return vm;
}

int main(int argc, char** argv) {
  boost::program_options::options_description desc;
  addOptions(desc);
  auto args = parseOptions(argc, argv, desc);

  spdlog::set_level(spdlog::level::info);

  // Create the high level scene description:
  auto meshFile = args["mesh-file"].as<std::string>();
  auto scene = makeCornellBoxScene(meshFile, args["box-only"].as<bool>());
  auto rngSeed = args["seed"].as<std::uint64_t>();
  if (args["render-mode"].as<std::string>() == "path-trace") {
    scene.pathTrace = std::make_unique<PathTraceSettings>();
    scene.pathTrace->sampler = xoshiro::Generator(rngSeed);
    scene.pathTrace->maxPathLength = args["max-path-length"].as<std::uint32_t>();
    scene.pathTrace->roulletteStartDepth = args["roulette-start-depth"].as<std::uint32_t>();
    scene.pathTrace->samplesPerPixel = args["samples"].as<std::uint32_t>();
    if (args.at("visualise").as<std::string>() != "rgb") {
      throw std::runtime_error("Running path-tracing without visualise=rgb is not advised.");
    }
  }

  // Scene needs to be converted into a compact representation
  // that can be shared (as far as possible) between Embree, CPU, and IPU
  // renders. Note: creation order is important in all cases because geomIDs
  // (Embree concept) are used to retrieve primitives during BVH traversal)
  // Mapping between materials and primitives also depends on a consistent order.
  SceneData data;

  // We need a compact representation for multiple meshes that we can transfer
  // to the device easily. Append all the triangle buffer indices and vertex
  // indices into unified arrays:
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
  }

  const bool ipuOnly = args["ipu-only"].as<bool>();

  // Initialise Embree:
  embree_utils::EmbreeScene embreeScene;

  // Create Embree and custom representations of all primitives:
  for (auto i = 0u; i < scene.meshes.size(); ++i) {
    auto& m = scene.meshes[i];
    data.geometry.emplace_back(i, GeomType::Mesh);
    embreeScene.addTriMesh(
      m.vertices,
      ConstArrayRef<std::uint32_t>::reinterpret(m.triangles.data(), m.triangles.size()));
  }

  for (auto i = 0u; i < scene.spheres.size(); ++i) {
    auto& s = scene.spheres[i];
    data.geometry.emplace_back(i, GeomType::Sphere);
    embreeScene.addSphere(s.centre, s.radius);
  }

  for (auto i = 0u; i < scene.discs.size(); ++i) {
    auto& d = scene.discs[i];
    data.geometry.emplace_back(i, GeomType::Disc);
    embreeScene.addDisc(d.c, d.n, d.r);
  }

  data.materials = scene.materials;
  data.matIDs = scene.matIDs;

  auto buildPrimitives = makeBuildPrimitivesForEmbree(data, scene);

  // Build our own BVH (still using Embree to build it):
  embree_utils::BvhBuilder builder(embreeScene.getDevice());
  builder.build(buildPrimitives);
  std::uint32_t maxDepth;

  // Convert our custom BVH into a compact form (i.e. convert the tree to a linear array):
  data.bvhNodes = buildCompactBvh(builder.getRoot(), builder.nodeCount(), maxDepth);
  buildPrimitives.clear(); // Embree is done with this now so free the space

  ipu_utils::logger()->debug("Max leaf depth in BVH: {}", maxDepth);

  // ===== Rendering: ======
  const auto visModeStr = args.at("visualise").as<std::string>();
  const auto visMode = visStrMap.at(visModeStr);
  const auto imageWidth = args["width"].as<std::int32_t>();
  const auto imageHeight = args["height"].as<std::int32_t>();
  const std::string outPrefix = "out_" + visModeStr + "_";
  scene.shadowRayOffset = args["ray-epsilon"].as<float>();

  // Test the SceneRef structure:
  SceneRef sceneRef {
    ConstArrayRef(data.geometry),
    ConstArrayRef(data.meshInfo),
    ConstArrayRef(data.meshTris),
    ConstArrayRef(data.meshVerts),
    ConstArrayRef(data.matIDs),
    ConstArrayRef(data.materials),
    ConstArrayRef(data.bvhNodes),
    maxDepth,
    scene.shadowRayOffset,
    rngSeed,
    (float)imageWidth,
    (float)imageHeight,
    args["samples"].as<std::uint32_t>(),
    args["max-path-length"].as<std::uint32_t>(),
    args["roulette-start-depth"].as<std::uint32_t>(),
    scene.pathTrace != nullptr
  };

  cv::Mat embreeImage(imageWidth, imageHeight, CV_32FC3);
  cv::Mat cpuImage(imageWidth, imageHeight, CV_32FC3);

  if (!ipuOnly) {
    // First create the same image using our custom built BVH and
    // custom intersection routines:
    auto rayStream = renderCPU(sceneRef, cpuImage, scene);
    auto hitCount = visualiseHits(rayStream, sceneRef, cpuImage, visMode);
    cv::imwrite(outPrefix + "cpu.exr", cpuImage);
    ipu_utils::logger()->debug("CPU reference hit count: {}", hitCount);

    // Now create reference image using embree:
    if (sceneRef.pathTrace) {
      ipu_utils::logger()->warn("Embree path trace not yet implemented.");
    } else {
      rayStream = renderEmbree(sceneRef, embreeScene, embreeImage);
      hitCount = visualiseHits(rayStream, sceneRef, embreeImage, visMode);
      cv::imwrite(outPrefix + "embree.exr", embreeImage);
      ipu_utils::logger()->debug("Embree hit count: {}", hitCount);
    }
  }

  // Now render on IPU:
  cv::Mat ipuImage(imageWidth, imageHeight, CV_32FC3);
  auto rayStream = renderIPU(sceneRef, ipuImage, scene.spheres, scene.discs, args);
  auto hitCount = visualiseHits(rayStream, sceneRef, ipuImage, visMode);
  cv::imwrite(outPrefix + "ipu.exr", ipuImage);
  ipu_utils::logger()->debug("IPU hit count: {}", hitCount);

  if (!ipuOnly) {
    // Compare IPU and CPU outputs:
    auto diff = ipuImage - cpuImage;
    auto sq = diff.mul(diff);
    auto mse = cv::mean(sq);
    ipu_utils::logger()->info("MSE IPU vs CPU result: {}", mse);

    // Note: Embree uses different algorithms internally so will never be an exact match:
    diff = ipuImage - embreeImage;
    sq = diff.mul(diff);
    mse = cv::mean(sq);
    ipu_utils::logger()->info("MSE IPU vs Embree result: {}", mse);
  }

  ipu_utils::logger()->info("Done.");
  return EXIT_SUCCESS;
}
