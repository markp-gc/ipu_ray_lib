// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// File contains some simple BRDFs and sampling functions.

#include <embree_utils/geometry.hpp>
#include <geometric_sampling.hpp>

// Sample from a diffuse material returning the sampled incoming ray direction.
// (Incoming because we are tracing from camera to light source but light
// travels the opposite way. Note: some path tracers use this convention, some don't).
inline
embree_utils::Vec3fa sampleDiffuse(const embree_utils::Vec3fa& normal, float u1, float u2) {
  using namespace embree_utils;
  // Get a coordinate system tangent to the hit point with z-axis parallel to the surface normal:
  auto [xBasis, yBasis, ignored] = normal.orthonormalSystem(); // zBasis == surface normal

  // Sample an incoming ray direction from diffuse BSDF:
  //const auto wiTangent = sampleHemisphere(u1, u2);
  const auto wiTangent = cosineSampleHemisphere(u1, u2);
  // Transform the sampled dir from tangent space to world coordinates
  // (i.e. multipying with the transpose tangent space basis matrix):
  auto wiWorld = Vec3fa(
    Vec3fa(xBasis.x, yBasis.x, normal.x).dot(wiTangent),
    Vec3fa(xBasis.y, yBasis.y, normal.y).dot(wiTangent),
    Vec3fa(xBasis.z, yBasis.z, normal.z).dot(wiTangent)
  );

  // Return the sampled incoming ray direction for next step:
  return wiWorld;
}

// Perfectly specular BRDF
inline
embree_utils::Vec3fa reflect(const embree_utils::Vec3fa& rayDir, const embree_utils::Vec3fa& normal) {
  auto cosTheta = rayDir.dot(normal);
  return (rayDir - (normal * (cosTheta * 2.f))).normalized();
}

inline float schlick(float cosTheta, float ri) {
    auto r0 = (1.f - ri) / (1.f + ri);
    r0 = r0 * r0;
    float base = 1.f - cosTheta;
    float base2 = base * base;
    float base5 = base2 * base * base2;
    return r0 + (1.f - r0) * base5;
}

inline
embree_utils::Vec3fa refract(const embree_utils::Vec3fa& dir, embree_utils::Vec3fa normal, float ndotr, float ri) {
  using namespace embree_utils;
  const auto cosTheta = -ndotr;
  Vec3fa rPerp = (dir + (normal * cosTheta)) * ri;
  Vec3fa rPar = normal * -std::sqrt(std::abs(1.f - rPerp.squaredNorm()));
  return rPerp + rPar;
}

inline
embree_utils::Vec3fa dielectric(const embree_utils::Ray& ray, embree_utils::Vec3fa normal,
             float ri, float u1) {
  if(normal.dot(ray.direction) > 0.f) {
    normal = -normal;
  } else {
    ri = 1.f / ri;
  }

  const float ndotr = normal.dot(ray.direction);
  const auto cost1 = -ndotr;
  const auto cost2 = 1.f - ri * ri * (1.f - cost1 * cost1);

  if (cost2 > 0.f && u1 > schlick(cost1, ri)) {
    return refract(ray.direction, normal, ndotr, ri);
  } else {
    return reflect(ray.direction, normal);
  }
}
