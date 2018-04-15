// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause
//
// This module represents a dataset sample.
//

#pragma once
#include <vector>

class Sample {
public:
  Sample(int totAttributes, int totClasses);
  std::vector<int64_t> inxValue_;
  std::vector<double> benefit_;
};