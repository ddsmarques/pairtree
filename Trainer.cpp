#include "Trainer.h"

#include "DataSet.h"
#include "DataSetBuilder.h"
#include "Tester.h"
#include "TrainReader.h"

void Trainer::train(std::string fileName) {
  TrainReader reader;
  std::shared_ptr<ConfigTrain> config = reader.read(fileName);

  DataSetBuilder builder;
  DataSet ds = builder.buildFromFile(config->dataSetFile);
  auto tree = config->tree->createTree(ds, config->configTree);
  tree->printNode();

  Tester tester;
  tester.test(tree, ds);
}

