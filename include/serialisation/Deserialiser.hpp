// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdint>
#include <type_traits>

// Class for de-serialising POD data and structures encoded
// by the serialiser (see Serialiser.hpp).
template <std::uint32_t BaseAlign = 16>
struct Deserialiser {
  constexpr std::uint32_t baseAlignment() const { return BaseAlign; }

  Deserialiser(std::uint8_t* bytes, std::size_t size) : end(bytes + size), ptr(bytes) {}

#ifndef __POPC__
  template <typename Alloc>
  Deserialiser(std::vector<std::uint8_t, Alloc>& v) : Deserialiser(v.data(), v.size()) {}
#endif

  template <typename T>
  std::uint32_t read(T& o) {
    constexpr auto align = alignof(T);
    constexpr auto size = sizeof(T);

    // Work out if the data would have been written with padding:
    const auto offset = (uintptr_t)(ptr + BaseAlign);
    const auto rem = offset % align;
    auto pad = 0u;
    if (rem) {
      pad = align - rem;
    }

    if (ptr + pad >= end) {
#ifndef __POPC__
      throw std::runtime_error("Deserialiser encountered end of byte stream.");
#endif
    }

    std::memcpy(&o, ptr + pad, sizeof(o));
    ptr += pad + sizeof(o);
    return pad + sizeof(o);
  };

  const std::uint8_t* getPtr() const { return ptr; }
  void skip(std::uint32_t count) { ptr += count; }

private:
  const std::uint8_t* const end;
  std::uint8_t* ptr;
};

// For non-fundamental types look for a user defined deserialise free function:
template <typename T, std::uint32_t BaseAlign, std::enable_if_t<!std::is_fundamental<T>::value>* = nullptr>
Deserialiser<BaseAlign>& operator >> (Deserialiser<BaseAlign>& d, T& v) {
  deserialise(d, v);
  return d;
}

// Provide an implementation for Eigen::half
template <std::uint32_t BaseAlign>
Deserialiser<BaseAlign>& operator >> (Deserialiser<BaseAlign>& d, half& v) {
  d.read(v);
  return d;
}

// For fundamental types call read directly:
template <typename T, std::uint32_t BaseAlign, std::enable_if_t<std::is_fundamental<T>::value>* = nullptr>
Deserialiser<BaseAlign>& operator >> (Deserialiser<BaseAlign>& d, T& v) {
  d.read(v);
  return d;
}
