#include <iostream>

#include "Trainer.h"

int main(int argc, char** argv) {
  Trainer trainer;
  trainer.train(argv[1]);
  return 0;
}
