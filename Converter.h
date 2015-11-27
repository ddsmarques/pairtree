// This module implements functions to help with the conversion from and to string.
// 
// Author: ddsmarques
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
