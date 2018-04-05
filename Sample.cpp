// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#include "Sample.h"

Sample::Sample(int totAttributes, int totClasses) {
  inxValue_.resize(totAttributes);
  benefit_.resize(totClasses);
}
