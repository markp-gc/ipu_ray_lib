// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <set>
#include <vector>

#include "ipu_utils.hpp"
#include <poputil/VarStructure.hpp>

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& s, const std::pair<T1, T2>& p) {
  s << "(" << p.first << ", " << p.second << ")";
  return s;
}

template <typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& v) {
  for (const auto& d : v) {
    s << d << " ";
  }
  return s;
}

template <typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<std::vector<T>>& vv) {
  for (const auto& v : vv) {
    if (v.empty()) {
      continue;
    }
    s << "[\n  ";
    for (const auto& d : v) {
      s << d << " ";
    }
    s << "\n]\n";
  }
  return s;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::set<T>& s) {
  for (auto& k : s) {
    os << k << " ";
  }
  return os;
}

inline void logTensorInfo(poplar::Graph& g, poplar::Tensor t) {
  ipu_utils::logger()->info("Shape: {}", t.shape());
  ipu_utils::logger()->info("Total elements: {}", t.numElements());
  ipu_utils::logger()->info("Innermost grouping: {}", poputil::detectInnermostGrouping(g, t));
  ipu_utils::logger()->info("Grouping dims: {}", poputil::detectDimGroupings(g, t));
  auto mapping = g.getTileMapping(t);
  auto tilesUsed = 0;
  for (auto& v : mapping) {
    if (!v.empty()) {
      tilesUsed += 1;
    }
  }
  ipu_utils::logger()->info("Tiles used: {}", tilesUsed);
}
