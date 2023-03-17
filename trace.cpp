// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// Main ray/path tracing programs for IPU, CPU, and Embree.

#include <app_utils.hpp>

std::vector<embree_utils::TraceResult> renderEmbree(const SceneRef& data, embree_utils::EmbreeScene& embreeScene, cv::Mat& image) {
  std::vector<embree_utils::TraceResult> rayStream(data.window.w * data.window.h);
  initPerspectiveRayStream(rayStream, image, data);
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

  ipu_utils::logger()->info("Embree Rendering started.");
  auto startTime = std::chrono::steady_clock::now();

  embreeScene.intersect(hitStream, image.rows);

  // For Embree we have to save hit records before shadow ray calcs
  // (because we want to use the streaming version of occlusion test
  // with the same ray stream for performance reasons):
  #pragma omp parallel for schedule(auto)
  for (auto i = 0u; i < hitStream.size(); ++i) {
    auto& rh = hitStream[i];
    if (std::isfinite(rh.ray.tfar)) {
      // Advance ray origin to hit point:
      rh.ray.org_x += rh.ray.dir_x * rh.ray.tfar;
      rh.ray.org_y += rh.ray.dir_y * rh.ray.tfar;
      rh.ray.org_z += rh.ray.dir_z * rh.ray.tfar;
    }
    rayStream[i].h = convertHitRecord(rh);
  }

  // Setup a shadow test:
  embree_utils::Vec3fa light(18, 257, -1060);
  #pragma omp parallel for schedule(auto)
  for (auto r = 0u; r < hitStream.size(); ++r) {
    auto& rh = hitStream[r];

    // Setup ray for an occlusion query (point ray at light):
    float dx = light.x - rh.ray.org_x;
    float dy = light.y - rh.ray.org_y;
    float dz = light.z - rh.ray.org_z;
    float d = std::sqrt(dx*dx + dy*dy + dz*dz);
    float norm = 1.f / d;
    rh.ray.dir_x = dx * norm;
    rh.ray.dir_y = dy * norm;
    rh.ray.dir_z = dz * norm;
    rh.ray.tnear = 0.f;
    rh.ray.tfar = d;

    // Offset the shadow ray to avoid self intersections:
    // const float absx = std::abs(rh.ray.org_x);
    // const float absy = std::abs(rh.ray.org_y);
    // const float absz = std::abs(rh.ray.org_z);
    // const float maxc = std::max(std::max(absx, absy), absz);
    // const float ndotd = (rh.ray.dir_x * rh.hit.Ng_x) + (rh.ray.dir_y * rh.hit.Ng_y) + ((rh.ray.dir_z * rh.hit.Ng_z));
    // float m = (1.f + maxc) * rayEpsilon * std::copysign(1.f, ndotd);
    // rh.ray.org_x += rh.hit.Ng_x * m;
    // rh.ray.org_y += rh.hit.Ng_y * m;
    // rh.ray.org_z += rh.hit.Ng_z * m;
    // Above works on CPU/IPU but not for Embree so use a more basic offset:
    rh.ray.org_x += rh.ray.dir_x * .005f;
    rh.ray.org_y += rh.ray.dir_y * .005f;
    rh.ray.org_z += rh.ray.dir_z * .005f;
  }
  embreeScene.occluded(hitStream, image.rows);

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
      // Not occluded:
      float norm = 1.f / std::sqrt(
        (rh.hit.Ng_x * rh.hit.Ng_x) +
        (rh.hit.Ng_y * rh.hit.Ng_y) +
        (rh.hit.Ng_z * rh.hit.Ng_z)
      );
      auto costh =
        norm * (rh.hit.Ng_x * rh.ray.dir_x) +
        norm * (rh.hit.Ng_y * rh.ray.dir_y) +
        norm * (rh.hit.Ng_z * rh.ray.dir_z);
      color += matRgb * costh;
    }
    rayStream[i].rgb = color;
  }

  auto endTime = std::chrono::steady_clock::now();
  ipu_utils::logger()->info("Embree Rendering ended.");

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
  hit.throughput = Vec3fa(1.f, 1.f, 1.f);
  Vec3fa color(0.f, 0.f, 0.f);

  for (auto i = 0u; i < scene.pathTrace->maxPathLength; ++i) {
    offsetRay(hit.r, hit.normal); // offset rays to avoid self intersection.
    // Reset ray limits for next bounce:
    hit.r.tMin = 0.f;
    hit.r.tMax = std::numeric_limits<float>::infinity();
    auto intersected = bvh.intersect(hit.r, primLookupFunc);

    if (intersected) {
      updateHit(intersected, hit);
      const auto& material = sceneRef.materials[sceneRef.matIDs[hit.geomID]];

      if (material.emissive) {
        color += hit.throughput * material.emission;
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
        hit.throughput *= material.albedo; // * (w / (Pi * pdf)); // PDF terms cancel for cosine weighted samples
        //throughput *= material.albedo * (wiTangent.z * 2.f); // Apply PDF for hemisphere samples (sampleDir is in tangent space so cos(theta) == z-coord).
      } else if (material.type == Material::Type::Specular) {
        hit.r.direction = reflect(hit.r.direction, hit.normal);
        hit.throughput *= material.albedo;
      } else if (material.type == Material::Type::Refractive) {
        float u1;
        #pragma omp critical(sample)
        { u1 = scene.pathTrace->sampler.uniform_0_1(); }
        const auto [dir, refracted] = dielectric(hit.r, hit.normal, material.ior, u1);
        hit.r.direction = dir;
        if (refracted) { hit.throughput *= material.albedo; }
      } else {
        // Mark an error:
        result.rgb *= std::numeric_limits<float>::quiet_NaN();
        hit.flags |= HitRecord::ERROR;
      }
    } else {
      hit.flags |= HitRecord::ESCAPED;
      break;
    }

    // Random stopping:
    if (i > scene.pathTrace->roulletteStartDepth) {
      float u1;
      #pragma omp critical(sample)
      {
        u1 = scene.pathTrace->sampler.uniform_0_1();
      }
      if (evaluateRoulette(u1, hit.throughput)) { break; }
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

    auto firstNormalIndex = 0u;
    auto numNormals = 0u;
    if (sceneRef.meshNormals.size()) {
      // If scene has normals assume every mesh has normals:
      firstNormalIndex = info.firstVertex;
      numNormals = info.numVertices;
    }
    meshes.emplace_back(
      embree_utils::Bounds3d(), // Don't actually need this bound box for rendering...
      ConstArrayRef(&sceneRef.meshTris[info.firstIndex], info.numTriangles),
      ConstArrayRef(&sceneRef.meshVerts[info.firstVertex], info.numVertices),
      ConstArrayRef(&sceneRef.meshNormals[firstNormalIndex], numNormals)
    );
  }

  std::vector<embree_utils::TraceResult> rayStream(sceneRef.window.w * sceneRef.window.h);
  initPerspectiveRayStream(rayStream, image, sceneRef);
  zeroRgb(rayStream);

  // Make a CompactBvh object for our custom CPU ray-tracer.
  // A CompactBvh wraps the scene ref BVH nodes:
  CompactBvh bvh(sceneRef.bvhNodes, sceneRef.maxLeafDepth);

  auto primLookup = [&](std::uint16_t geomID, std::uint32_t primID) {
    const auto& geom = sceneRef.geometry[geomID];
    return getPrimitive(geom, scene);
  };

  // Time just the intersections with the compact BVH:
  auto startTime = std::chrono::steady_clock::now();
  ipu_utils::logger()->info("CPU Rendering started.");

  if (scene.pathTrace) {
    for (auto s = 0u; s < scene.pathTrace->samplesPerPixel; ++s) {
      // Regenerate new camera rays at each sample step:
      initPerspectiveRayStream(rayStream, image, sceneRef, &scene.pathTrace->sampler);
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
        sceneRef.matIDs, sceneRef.materials,
        .05f, // ambient light factor
        *itr, primLookup, lightPos);
    }
  }

  ipu_utils::logger()->info("CPU Rendering finished.");
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
  std::vector<embree_utils::TraceResult> rayStream(sceneRef.window.w * sceneRef.window.h);
  initPerspectiveRayStream(rayStream, image, sceneRef);
  zeroRgb(rayStream);

  // This will be called as partial results are received from the IPU:
  IpuScene::RayCallbackFn rayCallback;
  IpuScene::RayCallbackFn* rayCallbackPtr = nullptr; // nullptr builds a renderer with no callback
  if (args["ipu-ray-callback"].as<bool>()) {
    rayCallback = [](std::size_t idx, const std::vector<embree_utils::TraceResult>& batch) {
      // here we just log but we could process the batch of results
      // e.g. to asynchronously update a render preview image.
      ipu_utils::logger()->debug("Application callback received batch {}", idx);
    };
    rayCallbackPtr = &rayCallback;
  }

  auto ipus = args["ipus"].as<std::uint32_t>();
  auto rpw = args["rays-per-worker"].as<std::size_t>();
  IpuScene ipuScene(spheres, discs, sceneRef, rayStream, rpw, rayCallbackPtr);
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

  // Hard code a large timeout. There is a
  // sync with the host ofter each ray batch but
  // a single batch could take a long time when a
  // large number of samples are used.
  ipu_utils::GraphManager().run(ipuScene,
                                {{"target.hostSyncTimeout", "10000"}});

  if (sceneRef.pathTrace) {
    scaleRgb(rayStream, 1.f / sceneRef.samplesPerPixel);
  }

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
  ("rays-per-worker", po::value<std::size_t>()->default_value(1), "Set the number of rays processed by each thread in each iteration. Lower values relieve I/O tile memory pressure.")
  ("width,w", po::value<std::int32_t>()->default_value(768), "Set rendered image width.")
  ("height,h", po::value<std::int32_t>()->default_value(432), "Set rendered image height.")
  ("crop", po::value<std::string>()->default_value(""),
   "String describing a window of the image to render. Format is wxh+c+r, "
   "where wxh is the width by height of window and +c+r specifies the column and row offset of the window.")
  ("anti-alias", po::value<float>()->default_value(.25f), "Width of anti-aliasing noise distribution in pixels.")
  ("mesh-file", po::value<std::string>()->default_value(std::string()),
                "Mesh file containing a scene to render. Format must be supported by libassimp "
                "(That library does not handle all formats well even if they are 'supported': "
                "consult the Blender export guide in the README. "
                "If no mesh file is specified the scene defaults to an built-in Cornell box scene.")
  ("load-normals", po::bool_switch()->default_value(false), "When loading a mesh file normals are ignored by default (to save on-chip memory). If you use this flag they will be loaded (and interpolated).")
  ("box-only", po::bool_switch()->default_value(false), "If rendering the built-in scene only render the original Cornell box without extra elements.")
  ("visualise", po::value<std::string>()->default_value("rgb"), "Choose the render output values to test/visualise. One of [rgb, normal, hitpoint, tfar, color, id]")
  ("render-mode", po::value<std::string>()->default_value("path-trace"), "Choose type of render from [shadow-trace, path-trace]. To see result set visualise=rgb")
  ("max-path-length", po::value<std::uint32_t>()->default_value(12), "Max path length for path tracing.")
  ("roulette-start-depth", po::value<std::uint32_t>()->default_value(5), "Path length after which rays can be randomly terminated with prob. inversely proportional to their throughput.")
  ("samples", po::value<std::uint32_t>()->default_value(256), "Number of samples per pixel for path tracing.")
  ("seed", po::value<std::uint64_t>()->default_value(1442), "RNG seed.")
  ("ipu-only", po::bool_switch()->default_value(false), "Only render on IPU (e.g. if you don't want to wait for slow CPU path tracing).")
  ("ipu-ray-callback", po::bool_switch()->default_value(false), "Retrieve partial results directly from the IPU during renderering via callback mechanism. "
                                                                "By default the results are read from DRAM on one go at the end of renderering.")
  ("log-level", po::value<std::string>()->default_value("info"),
  "Set the log level to one of the following: 'trace', 'debug', 'info', 'warn', 'err', 'critical', 'off'.");
}

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

  if (vm.at("mesh-file").as<std::string>().empty() && vm.at("load-normals").as<bool>()) {
    throw std::runtime_error("Option 'load-normals' is not valid without the 'mesh-file' option.");
  }

  po::notify(vm);
  return vm;
}

int main(int argc, char** argv) {
  boost::program_options::options_description desc;
  addOptions(desc);
  boost::program_options::variables_map args;
  try {
    args = parseOptions(argc, argv, desc);
    setupLogging(args);
  } catch (const std::exception& e) {
    ipu_utils::logger()->info("Exiting after: {}.", e.what());
    return EXIT_FAILURE;
  }

  // Log size info for various types, useful during memory optimisation:
  ipu_utils::logger()->trace("HitRecord size: {}", sizeof(embree_utils::HitRecord));
  ipu_utils::logger()->trace("HitRecord align: {}", alignof(embree_utils::HitRecord));
  ipu_utils::logger()->trace("TraceResult size: {}", sizeof(embree_utils::TraceResult));
  ipu_utils::logger()->trace("TraceResult align: {}", alignof(embree_utils::TraceResult));
  ipu_utils::logger()->trace("CompactBVH2Node size: {}", sizeof(CompactBVH2Node));
  ipu_utils::logger()->trace("CompactBVH2Node align: {}", alignof(CompactBVH2Node));
  ipu_utils::logger()->trace("Ray size: {}", sizeof(embree_utils::Ray));
  ipu_utils::logger()->trace("Ray align: {}", alignof(embree_utils::Ray));
  ipu_utils::logger()->trace("RayShearParams size: {}", sizeof(RayShearParams));
  ipu_utils::logger()->trace("RayShearParams align: {}", alignof(RayShearParams));

  // ===== Scene setup: ======

  // Load or build a scene:
  auto scene = buildSceneDescription(args);

  // Convert scene into efficient representations for rendering:
  auto [customScene, embreeScene] = buildSceneData(scene);

  // Get cropped window size:
  const auto imageWidth = args["width"].as<std::int32_t>();
  const auto imageHeight = args["height"].as<std::int32_t>();
  auto crop = parseCropString(args["crop"].as<std::string>());
  auto window = crop.value_or(CropWindow{imageWidth, imageHeight, 0, 0}); // (Set window to whole image if crop wasn't specified)
  ipu_utils::logger()->info("Rendering window: width: {}, height: {}, start col: {}, start row: {}", window.w, window.h, window.c, window.r);

  // The SceneRef wraps the dynamic arrays from the custom scene represenation in data structures that
  // can be backed by either dynamic (for CPU) or static (for IPU) arrays. This allows the CPU code to
  // be almost identical to IPU code which makes development and debugging quicker:
  SceneRef sceneRef {
    ConstArrayRef(customScene.geometry),
    ConstArrayRef(customScene.meshInfo),
    ConstArrayRef(customScene.meshTris),
    ConstArrayRef(customScene.meshVerts),
    ConstArrayRef(customScene.meshNormals),
    ConstArrayRef(customScene.matIDs),
    ConstArrayRef(customScene.materials),
    ConstArrayRef(customScene.bvhNodes),
    customScene.bvhMaxDepth,
    args["seed"].as<std::uint64_t>(),
    (float)imageWidth,
    (float)imageHeight,
    scene.camera.horizontalFov,
    args["anti-alias"].as<float>(),
    window,
    args["samples"].as<std::uint32_t>(),
    args["max-path-length"].as<std::uint32_t>(),
    args["roulette-start-depth"].as<std::uint32_t>(),
    scene.pathTrace != nullptr
  };

  // ===== Rendering: ======

  const auto visModeStr = args.at("visualise").as<std::string>();
  const auto visMode = visStrMap.at(visModeStr);
  const std::string outPrefix = "out_" + visModeStr + "_";

  cv::Mat embreeImage(imageHeight, imageWidth, CV_32FC3);
  cv::Mat cpuImage(imageHeight, imageWidth, CV_32FC3);

  const bool ipuOnly = args["ipu-only"].as<bool>();
  if (!ipuOnly) {
    // First create the same image using our custom built BVH and
    // custom intersection routines:
    auto rayStream = renderCPU(sceneRef, cpuImage, scene);
    auto hitCount = visualiseHits(rayStream, sceneRef, cpuImage, visMode);
    cv::imwrite(outPrefix + "cpu.exr", cpuImage);
    ipu_utils::logger()->debug("CPU reference hit count: {}", hitCount);

    // Now create reference image using Embree:
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
  cv::Mat ipuImage(imageHeight, imageWidth, CV_32FC3);
  auto rayStream = renderIPU(sceneRef, ipuImage, scene.spheres, scene.discs, args);
  auto hitCount = visualiseHits(rayStream, sceneRef, ipuImage, visMode);
  cv::imwrite(outPrefix + "ipu.exr", ipuImage);
  ipu_utils::logger()->debug("IPU hit count: {}", hitCount);

  // ===== Testing: ======

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
