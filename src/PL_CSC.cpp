// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#include <iostream>

#include "Trainer.h"

int main(int argc, char** argv) {
  Trainer trainer;
  trainer.train(argv[1]);
  return 0;
}
