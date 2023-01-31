// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// IPU C++ codelets (kernels) can not use dynamic memory allocation.
// In order to maintain consistent code between host C++ and IPU C++ code
// this file contains some array reference objects that do not allocate any memory
// and can be used to provide a consistent interface for dynamic arrays
// (on CPU) or static arrays (on IPU).

#pragma once

#include <cstdint>
#include <algorithm>

#ifndef __IPU__
#include <vector>
#endif

template <typename T>
struct ConstArrayRef {

  ConstArrayRef() : data(nullptr), len(0) {}

  ConstArrayRef(const T* ptr, std::uint32_t numElements)
    : data(ptr), len(numElements) {}

#ifndef __IPU__
  ConstArrayRef(const std::vector<T>& v)
    : data(v.data()), len(v.size()) {}
#endif

  // Reinterpret an array ref of the template type from a byte pointer:
  template <typename UnderlyingType>
  static ConstArrayRef reinterpret(const UnderlyingType* data, std::uint32_t size) {
    // Note: this calculation only works for packed storage:
    return ConstArrayRef(reinterpret_cast<const T*>(data), (size * sizeof(UnderlyingType))/sizeof(T));
  }

  const T& operator[] (std::uint32_t i) const {
    return data[i];
  }

  std::uint32_t size() const { return len; }

  const T& front() const { return data[0]; }
  const T& back() const { return data[len - 1]; }

  const T* begin() { return data; }
  const T* end() { return data + len; }
  const T* cbegin() const { return data; }
  const T* cend()   const { return data + len; }

private:
  const T* const data;
  const std::uint32_t len;
};

template <typename T>
struct ArrayRef {

  ArrayRef() : data(nullptr), len(0) {}

  ArrayRef(T* ptr, std::uint32_t numElements)
    : data(ptr), len(numElements) {}

#ifndef __IPU__
  ArrayRef(std::vector<T>& v)
    : data(v.data()), len(v.size()) {}
#endif

  // Reinterpret an array ref of the template type from a byte pointer:
  template <typename UnderlyingType>
  static ArrayRef reinterpret(UnderlyingType* data, std::uint32_t size) {
    // Note: this calculation only works for packed storage:
    return ArrayRef(reinterpret_cast<T*>(data), (size * sizeof(UnderlyingType))/sizeof(T));
  }

  T& operator[] (std::uint32_t i) {
    return data[i];
  }

  std::uint32_t size() const { return len; }

  T& front() { return data[0]; }
  T& back() { return data[len - 1]; }

  T* begin() { return data; }
  T* end() { return data + len; }
  const T* cbegin() const { return data; }
  const T* cend()   const { return data + len; }

private:
  T* const data;
  const std::uint32_t len;
};

// Stack with a fixed maximum capacity with similar interface
// to std::vector for quick porting of existing code:
template <typename T, std::size_t Capacity>
class ArrayStack {
    std::size_t nexti;
    T store[Capacity];

public:
    ArrayStack() : nexti(0) {}
    ~ArrayStack() {}

    bool full() const { return nexti == Capacity; }
    constexpr std::size_t capacity() const { return Capacity; }
    bool empty() const { return nexti == 0; }
    std::size_t size() const { return nexti; }
    void push_back(const T& value) {
      store[nexti] = value; nexti += 1;
    }
    void pop_back() { nexti -= 1; }
    const T& back() const { return store[nexti - 1]; }
    T& back() { return store[nexti - 1]; }
    void clear() { nexti = 0; }
    const T& operator[] (std::size_t i) const { return store[i]; }
};

template <typename T>
class WrappedArray {
    std::size_t nexti;
    ArrayRef<T> store;

public:
    WrappedArray(const ArrayRef<T>& array)
      : nexti(0),
        store(array)
    {}

    ~WrappedArray() {}

    bool full() const { return nexti == store.size(); }
    constexpr std::size_t capacity() const { return store.size(); }
    bool empty() const { return nexti == 0; }
    std::size_t size() const { return nexti; }
    void push_back(const T& value) {
      store[nexti] = value;
      nexti += 1;
    }
    void pop_back() { nexti -= 1; }
    const T& back() const { return store[nexti - 1]; }
    T& back() { return store[nexti - 1]; }
    void clear() { nexti = 0; }
    const T& operator[] (std::size_t i) const { return store[i]; }
    T& operator[] (std::size_t i) { return store[i]; }
    void skip(std::size_t n = 1) { nexti += n; }
};
