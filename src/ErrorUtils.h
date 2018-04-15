// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause
//
// This module implements functions to help dealing with error messages.
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