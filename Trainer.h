#pragma once
#include <string>

#include "PairTree.h"
#include "TrainReader.h"
#include "Tester.h"

class Trainer {
public:
  void train(std::string fileName);

private:
  struct TreeResult {
    long double score;
    long double savings;
    int64_t size;
    int64_t seconds;
    std::vector<std::vector<long double>> alphaXsamples;
  };

  TreeResult execRandomSplit(std::shared_ptr<ConfigTrain> config,
                             DataSet& trainDS, DataSet& testDS,
                             int treeInx);
  TreeResult execSplit(std::shared_ptr<ConfigTrain> config,
                      DataSet& trainDS, int treeInx);
  void getRandomSplit(DataSet& trainDS, DataSet& testDS, double ratio);
  void getSplit(DataSet& originalDS, DataSet& currTrain,
                DataSet& currTest, std::string fileName,
                int fold);
  void loadSplit(DataSet& originalDS, DataSet& current, std::string fileName);
  TreeResult runTree(std::shared_ptr<ConfigTrain>& config, int treeInx,
                     DataSet& trainDS, DataSet& testDS);
  TreeResult runPairTree(std::shared_ptr<PairTree> pairTree,
                         std::shared_ptr<ConfigPairTree> config,
                         DataSet& trainDS, DataSet& testDS);

  std::string outputFolder_;
};
