// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

// Class for serialising POD data and structures.

#include <cstdint>
#include <type_traits>

#include <boost/align/aligned_allocator.hpp>
#include <Eigen/Dense>
using half = Eigen::half;

// Utility for serialising data. This object makes sure that
// the alignment of all serialised objects is repsected within
// the serialised bytes stream under the assumption that the
// first byte stored is aligned with BaseAlign.
//
// The purpose of aligment within the serialised representation is
// so that objects (especially arrays) in the archive can be accessed
// by reinterpretting pointers without  deserialisation.
template <std::uint32_t BaseAlign = 16>
struct Serialiser {

  constexpr std::uint32_t baseAlignment() const { return BaseAlign; }

  Serialiser(std::uint32_t capacity) {
    bytes.reserve(capacity);
  }

  template <typename T>
  std::uint32_t write(const T& o) {
    constexpr auto align = alignof(T);
    constexpr auto size = sizeof(T);

    // Work out if we need padding:
    const auto currentByteOffset = BaseAlign + bytes.size();
    const auto rem = currentByteOffset % align;
    auto pad = 0u;
    if (rem) {
      pad = align - rem;
    }

    bytes.resize(bytes.size() + pad + sizeof(o));
    std::memcpy(&bytes.back() + 1 - sizeof(o), &o, sizeof(o));
    return pad + sizeof(o);
  };

  std::vector<std::uint8_t,
              boost::alignment::aligned_allocator<std::uint8_t, BaseAlign>> bytes;
};

// For non-fundamental types look for a user defined serialise free function:
template <typename T, std::uint32_t BaseAlign, std::enable_if_t<!std::is_fundamental<T>::value>* = nullptr>
Serialiser<BaseAlign>& operator << (Serialiser<BaseAlign>& s, const T& v) {
  serialise(s, v);
  return s;
}

// Provide an implementation for Eigen::half
template <std::uint32_t BaseAlign>
Serialiser<BaseAlign>& operator << (Serialiser<BaseAlign>& s, const half& v) {
  s.write(v);
  return s;
}

// For fundamental types call write directly:
template <typename T, std::uint32_t BaseAlign, std::enable_if_t<std::is_fundamental<T>::value>* = nullptr>
Serialiser<BaseAlign>& operator << (Serialiser<BaseAlign>& s, const T& v) {
  s.write(v);
  return s;
}
