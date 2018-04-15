// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause
//
// This module implements functions to help with the conversion from and to string.
//

#pragma once
#include <string>

class Converter {
public:
  static bool isInteger(const std::string& str);
  static bool isDouble(const std::string& str);
  template <typename T>
  static T fromString(const std::string& str);
};
