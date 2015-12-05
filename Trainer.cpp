#include "Trainer.h"

#include "DataSet.h"
#include "DataSetBuilder.h"
#include "Tester.h"
#include "TrainReader.h"

void Trainer::train(std::string fileName) {
  TrainReader reader;
  std::shared_ptr<ConfigTrain> config = reader.read(fileName);

  Tester tester;
  DataSetBuilder builder;
  DataSet ds = builder.buildFromFile(config->dataSetFile);
  for (int64_t i = 0; i < config->configTrees.size(); i++) {
    auto tree = config->trees[i]->createTree(ds, config->configTrees[i]);
    tree->printNode();
    tester.test(tree, ds);
  }
}

