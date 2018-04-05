// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#pragma once

class CompareUtils {
public:
  static int compare(const long double& a, const long double& b, long double eps = 1e-9);
};