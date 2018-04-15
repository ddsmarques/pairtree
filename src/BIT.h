// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#pragma once
#include <vector>


class BIT {
public:
  BIT(int64_t N);
  void update(int64_t inx, double value);
  double get(int64_t inx);

private:
  int64_t N_;
  std::vector<double> tree_;
};
