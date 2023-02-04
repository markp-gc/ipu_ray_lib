// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

// Utilities for building applications ontop of ipu_ray_lib.

#include <boost/program_options.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <map>
#include <string>

#include <math/sincos.hpp>
#include <IpuScene.hpp>
#include <CompactBvhBuild.hpp>
#include <CompactBvh.hpp>
#include <BxDF.hpp>
#include <Render.hpp>
#include <scene_utils.hpp>
#include <embree_utils/bvh.hpp>
#include <embree_utils/EmbreeScene.hpp>
#include <embree_utils/geometry.hpp>

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

void initPerspectiveRayStream(std::vector<embree_utils::TraceResult>& rayStream,
                              const cv::Mat& image,
                              const SceneRef& data,
                              xoshiro::Generator* gen = nullptr);

void zeroRgb(std::vector<embree_utils::TraceResult>& rayStream);

void scaleRgb(std::vector<embree_utils::TraceResult>& rayStream, float scale);

unsigned visualiseHits(const std::vector<embree_utils::TraceResult>& rayStream,
                       const SceneRef& data, cv::Mat& image, VisualiseMode mode);

const Primitive* getPrimitive(GeomRef geom, const SceneDescription& scene);

void setupLogging(const boost::program_options::variables_map& args);

std::optional<CropWindow> parseCropString(const std::string& cropFmt);

std::unique_ptr<PathTraceSettings> makePathTraceSettings(const boost::program_options::variables_map& args);

// Load or build the scene description (depending on args).
// This is a high level description (i.e. in case of loading
// from file the scne is imported into data structures that
// mirror the file import format).
SceneDescription buildSceneDescription(const boost::program_options::variables_map& args);

// Build efficient scene representations for both Embree and our custom CPU/IPU renderers/
//
// The scene description needs to be converted into a compact representation
// that can be shared (as far as possible) between Embree, CPU, and IPU
// renders.
//
// Note: creation order is important in all cases because geomIDs
// (Embree concept) are used to retrieve primitives during BVH traversal)
// Mapping between materials and primitives also depends on a consistent order.
std::pair<SceneData, embree_utils::EmbreeScene> buildSceneData(const SceneDescription& scene);
