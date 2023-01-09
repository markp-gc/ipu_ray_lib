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
  auto cost = rayDir.dot(normal);
  return (rayDir - (normal * (cost * 2.f))).normalized();
}

// Glass/refractive BRDF - we use the vector version of Snell's law and Fresnel's law
// to compute the outgoing reflection and refraction directions and probability weights.
// Returns true if the ray was refracted.
inline
embree_utils::Vec3fa refract(const embree_utils::Ray& ray, embree_utils::Vec3fa normal,
             float ri, float u1) {
  auto r0 = (1.f - ri)/(1.f + ri);
  r0 = r0 * r0;
  if(normal.dot(ray.direction) > 0.f) { // we're inside the medium
    normal = -normal;
  } else {
    ri = 1.f / ri;
  }
  auto cost1 = -normal.dot(ray.direction); // cosine of theta_1
  auto cost2 = 1.f - ri * ri * (1.f - cost1 * cost1); // cosine of theta_2
  auto schlickBase = 1.f - cost1;
  auto schlickBase2 = schlickBase * schlickBase;
  auto prob = r0 + (1.f - r0) * (schlickBase2 * schlickBase * schlickBase2); // Schlick-approximation
  if (cost2 > 0.f && u1 > prob) {
    // refraction direction
    return ((ray.direction * ri) + (normal * (ri * cost1 - sqrtf(cost2)))).normalized();
  } else {
    // reflection direction
    return (ray.direction + normal * (cost1 * 2.f)).normalized();
  }
}
