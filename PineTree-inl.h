#pragma once
#include "gurobi_c++.h"

template <typename T>
void PineTree::initMultidimension(std::vector<T>& v, std::vector<int64_t> dim) {
  if (dim.size() == 0) {
    return;
  }
  v.resize(dim[0]);
  dim.erase(dim.begin());
  for (int i = 0; i < v.size(); i++) {
    initMultidimension(v[i], dim);
  }
}

template <typename T>
void PineTree::initMultidimension(T& v, std::vector<int64_t> dim) {
  return;
}
