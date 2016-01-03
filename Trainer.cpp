#include "Trainer.h"

#include "DataSet.h"
#include "DataSetBuilder.h"
#include "Tester.h"
#include "TrainReader.h"

#include <ctime>
#include <iomanip>
#include <fstream>

void Trainer::train(std::string fileName) {
  TrainReader reader;
  std::shared_ptr<ConfigTrain> config = reader.read(fileName);

  // Creates a folder appending the time to the end
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::stringstream buffer;
  buffer << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
  std::string folderName = config->outputFolder + config->name + "_" + buffer.str() + "\\";
  std::string cmd = "mkdir " + folderName;
  system(cmd.c_str());

  std::string outputFileName = folderName + "output.txt";
  std::ofstream outputFile;
  outputFile.open(outputFileName, std::ofstream::out);
  outputFile.close();

  Tester tester;
  DataSetBuilder builder;
  DataSet ds = builder.buildFromFile(config->dataSetFile, config->classColStart);
  for (int64_t i = 0; i < config->configTrees.size(); i++) {
    std::shared_ptr<DecisionTreeNode> tree = config->trees[i]->createTree(ds, config->configTrees[i]);
    ds.printTree(tree, outputFileName);
    tester.test(tree, ds, outputFileName);
  }
}

