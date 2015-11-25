// This module implements functions to help dealing with error messages.
// 
// Author: ddsmarques
//
#pragma once
#include <string>

namespace ErrorUtils {
/**
* Throw an exception if the first argument is false.  The exception
* message will contain the argument string as well as any passed-in
* arguments to enforce, formatted using folly::to<std::string>.
*/
  void enforce(bool v, std::string msg = "");
}