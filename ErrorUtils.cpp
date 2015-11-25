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