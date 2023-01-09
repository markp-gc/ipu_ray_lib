// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// This file contains fundamental ray-tracing geometric data types.
// The data types are specifically intended to be Embree compatible/interoperable.

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace embree_utils {

static constexpr double doublePi = 3.14159265358979323846264338327950288;
static constexpr float Pi = doublePi;
static constexpr float InvPi = 1.0 / doublePi;
static constexpr float Piby2 = (float)(doublePi / 2.0);
static constexpr float Piby4 = (float)(doublePi / 4.0);

struct __attribute__ ((aligned (16))) Vec3fa {
  Vec3fa() {}
  Vec3fa(float _v) : x(_v), y(_v), z(_v) {}
  Vec3fa(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

  union {
    struct { float x, y, z; };
    float c[3];
  };

  float& operator[](std::uint32_t i) {
    return c[i];
  }

  const float& operator[](std::uint32_t i) const {
    return c[i];
  }

  void normalize() {
    float scale = 1.f / std::sqrt(x*x + y*y + z*z);
    x *= scale;
    y *= scale;
    z *= scale;
  }

  Vec3fa operator - () const {
    return Vec3fa(-x, -y, -z);
  }

  Vec3fa operator - (const Vec3fa& v) const {
    return Vec3fa(x - v.x, y - v.y, z - v.z);
  }

  Vec3fa operator + (const Vec3fa& v) const {
    return Vec3fa(x + v.x, y + v.y, z + v.z);
  }

  Vec3fa operator - (float f) const {
    return Vec3fa(x - f, y - f, z - f);
  }

  Vec3fa operator + (float f) const {
    return Vec3fa(x + f, y + f, z + f);
  }

  Vec3fa operator * (float f) const {
    return Vec3fa(x * f, y * f, z * f);
  }

  Vec3fa operator * (const Vec3fa& v) const {
    return Vec3fa(x * v.x, y * v.y, z * v.z);
  }

  Vec3fa& operator += (const Vec3fa& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }

  Vec3fa& operator -= (const Vec3fa& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }

  Vec3fa& operator *= (const Vec3fa& v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
  }

  Vec3fa min(const Vec3fa& other) const {
    return Vec3fa(
      std::min(x, other.x),
      std::min(y, other.y),
      std::min(z, other.z));
  }

  Vec3fa max(const Vec3fa& other) const {
    return Vec3fa(
      std::max(x, other.x),
      std::max(y, other.y),
      std::max(z, other.z));
  }

  std::uint32_t maxi() const {
    // Return index of maximum component:
    if (x < y) {
      return x < z ? 0 : 2;
    }
    return y < z ? 1 : 2;
  }

  float maxc() const {
    return c[maxi()];
  }

  Vec3fa abs() const {
    return Vec3fa(std::abs(x), std::abs(y), std::abs(z));
  }

  Vec3fa permute(std::uint32_t ix, std::uint32_t iy, std::uint32_t iz) const {
    return Vec3fa(c[ix], c[iy], c[iz]);
  }

  float squaredNorm() const {
    return x*x + y*y + z*z;
  }

  Vec3fa normalized() const { return *this * (1.f/sqrtf(squaredNorm())); }

  float dot(const Vec3fa &b) const { return x*b.x + y*b.y + z*b.z; }

  Vec3fa cross(const Vec3fa &v) const {
    return Vec3fa(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
  }

  std::tuple<Vec3fa, Vec3fa, Vec3fa> orthonormalSystem() const {
    Vec3fa v2;
    const Vec3fa v1abs = this->abs();
    const Vec3fa v1sq = (*this) * (*this);
    if (v1abs.x > v1abs.y) {
      float invLen = 1.f / std::sqrt(v1sq.x + v1sq.z);
      v2 = Vec3fa(-this->z * invLen, 0.f, this->x * invLen);
    } else {
      float invLen = 1.f / std::sqrt(v1sq.y + v1sq.z);
      v2 = Vec3fa(0.f, this->z * invLen, -this->y * invLen);
    }
    return std::make_tuple(v2, this->cross(v2), *this);
  }

  bool isZero() const { return x == 0.f && y == 0.f && z == 0.f; }
  bool isNonZero() const { return x != 0.f || y != 0.f || z != 0.f; }
};

struct Bounds3d {
  Bounds3d(bool) {
    // Overload to skip default init. Used to preseve contents on references.
  }

  Bounds3d()
    : min(std::numeric_limits<float>::infinity()), max(-std::numeric_limits<float>::infinity()) {}
  Bounds3d(const Vec3fa& _min, const Vec3fa& _max) : min(_min), max(_max) {}

  Vec3fa centroid() const {
    return (max + min) * .5f;
  }

  void operator += (const Bounds3d& other) {
    min.x = std::min(min.x, other.min.x);
    min.y = std::min(min.y, other.min.y);
    min.z = std::min(min.z, other.min.z);
    max.x = std::max(max.x, other.max.x);
    max.y = std::max(max.y, other.max.y);
    max.z = std::max(max.z, other.max.z);
  }

  void operator += (const Vec3fa& v) {
    min.x = std::min(min.x, v.x);
    min.y = std::min(min.y, v.y);
    min.z = std::min(min.z, v.z);
    max.x = std::max(max.x, v.x);
    max.y = std::max(max.y, v.y);
    max.z = std::max(max.z, v.z);
  }

  Vec3fa min;
  Vec3fa max;
};

struct PixelCoord {
  PixelCoord() // Constructs (probably) invalid pixel coords:
    : u(-std::numeric_limits<float>::infinity()),
      v(-std::numeric_limits<float>::infinity()) {}

  PixelCoord(std::uint32_t _u, std::uint32_t _v)
    : u(_u), v(_v) {}

  float u;
  float v;
};

struct __attribute__ ((aligned (16))) Ray {
  Ray() {}
  Ray(const Vec3fa& o, const Vec3fa& d) :
    origin(o),
    tMin(0.f),
    direction(d),
    tMax(std::numeric_limits<float>::infinity())
  {}

  embree_utils::Vec3fa origin;
  float tMin;
  embree_utils::Vec3fa direction;
  float tMax;
};

struct HitRecord {
  static constexpr std::uint16_t InvalidGeomID = std::numeric_limits<std::uint16_t>::max();
  static constexpr std::uint32_t InvalidPrimID = std::numeric_limits<std::uint32_t>::max();

  HitRecord() {}
  HitRecord(const Vec3fa& origin, const Vec3fa& dir)
    : r(origin, dir),
      primID(InvalidPrimID),
      normal(0.f, 0.f, 1.f), // Match Embree init
      geomID(InvalidGeomID)
  {}

  Ray r;
  std::uint32_t primID;
  Vec3fa normal;
  std::uint16_t geomID;
};

struct TraceResult {
  TraceResult() {}
  TraceResult(const HitRecord& hit, const PixelCoord& uv) : h(hit), p(uv), rgb(0.f, 0.f, 0.f) {}
  HitRecord h;
  embree_utils::PixelCoord p;
  embree_utils::Vec3fa rgb;
};

} // end namespace embree_utils
