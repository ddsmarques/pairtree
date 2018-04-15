// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#include "ErrorUtils.h"
#include <iostream>

namespace ErrorUtils {
void enforce(bool v, std::string msg) {
  if (!v) {
    std::cout << "Error: " << msg << std::endl;
    abort();
  }
}
}