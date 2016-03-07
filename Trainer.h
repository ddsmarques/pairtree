#pragma once
#include <string>

#include "TrainReader.h"

class Trainer {
public:
  void train(std::string fileName);

private:
  void createTrainTestDS(std::shared_ptr<ConfigTrain> config);
  DataSet trainDS_;
  DataSet testDS_;
};
