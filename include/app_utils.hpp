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

std::vector<RTCBuildPrimitive>
makeBuildPrimitivesForEmbree(const SceneData& data, const SceneDescription& scene);

void setupLogging(const boost::program_options::variables_map& args);

boost::program_options::variables_map
parseOptions(int argc, char** argv, boost::program_options::options_description& desc);

std::optional<CropWindow> parseCropString(const std::string& cropFmt);
