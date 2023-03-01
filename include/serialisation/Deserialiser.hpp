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

  // Read an object respecting the alignment that Serialiser would have
  // applied w.r.t the base alignment:
  template <typename T>
  std::uint32_t read(T& o) {
    // Work out if the data would have been written with padding:
    const auto offset = (uintptr_t)(ptr + BaseAlign);
    const auto rem = offset % alignof(T);
    auto pad = 0u;
    if (rem) {
      pad = alignof(T) - rem;
    }

    ptr += pad;
    checkForEndOfData(sizeof(T));

    std::memcpy(&o, ptr, sizeof(T));
    ptr += sizeof(T);
    return pad + sizeof(T);
  };

  // Read bytes directly from current point in buffer:
  void read(std::uint8_t* dst, std::uint64_t size) {
    checkForEndOfData(size);
    std::memcpy(dst, ptr, size);
    ptr += size;
  }

  const std::uint8_t* getPtr() const { return ptr; }
  void skip(std::uint64_t count) {
    checkForEndOfData(count);
    ptr += count;
  }

private:
  const std::uint8_t* const end;
  std::uint8_t* ptr;

  void checkForEndOfData(std::uint64_t readSize) {
    auto newPtr = ptr + readSize;
    if (newPtr > end) {
#ifdef __POPC__
      assert(newPtr <= end);
#else
      std::cerr << "invalid ptr: " << (std::uintptr_t)newPtr << "\n";
      throw std::runtime_error("Deserialiser encountered end of byte stream.");
#endif
    }
  }
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
