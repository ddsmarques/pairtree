#pragma once
#include <string>

#include "TrainReader.h"

class Trainer {
public:
  void train(std::string fileName);

private:
  void getRandomSplit(DataSet& trainDS, DataSet& testDS, double ratio);
  void getSplit(DataSet& originalDS, DataSet& currTrain,
                DataSet& currTest, std::string fileName,
                int fold);
  void loadSplit(DataSet& originalDS, DataSet& current, std::string fileName);
  double runTree(std::shared_ptr<ConfigTrain>& config, int treeInx,
                 DataSet& trainDS, DataSet& testDS);

  std::string outputFolder_;
};
