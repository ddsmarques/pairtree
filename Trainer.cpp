#include "Trainer.h"

#include "DataSet.h"
#include "DataSetBuilder.h"
#include "Tester.h"
#include "TrainReader.h"

#include <fstream>

void Trainer::train(std::string fileName) {
  TrainReader reader;
  std::shared_ptr<ConfigTrain> config = reader.read(fileName);
  std::string outputFileName = config->outputFolder + config->name + "_output.txt";
  std::ofstream outputFile;
  outputFile.open(outputFileName, std::ofstream::out);
  outputFile.close();

  Tester tester;
  DataSetBuilder builder;
  DataSet ds = builder.buildFromFile(config->dataSetFile);
  for (int64_t i = 0; i < config->configTrees.size(); i++) {
    std::shared_ptr<DecisionTreeNode> tree = config->trees[i]->createTree(ds, config->configTrees[i]);
    ds.printTree(tree, outputFileName);
    tester.test(tree, ds, outputFileName);
  }
}

