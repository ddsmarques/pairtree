#include "Trainer.h"

#include "DataSet.h"
#include "DataSetBuilder.h"
#include "Tester.h"
#include "TrainReader.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <fstream>

void Trainer::train(std::string fileName) {
  TrainReader reader;
  std::shared_ptr<ConfigTrain> config = reader.read(fileName);

  // Creates a folder appending the time to the end
  std::stringstream buffer;
  std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  buffer << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");
  std::string folderName = config->outputFolder + config->name + "_" + buffer.str() + "\\";
  std::string cmd = "mkdir " + folderName;
  system(cmd.c_str());

  std::string outputFileName = folderName + "output.txt";
  std::ofstream outputFile;
  outputFile.open(outputFileName, std::ofstream::out);
  outputFile.close();

  std::string timeFileName = folderName + "time_log.txt";
  std::ofstream timeFile;
  timeFile.open(timeFileName, std::ofstream::out);
  timeFile << "Test name: " << config->name << std::endl;
  timeFile.close();

  Tester tester;
  DataSetBuilder builder;
  DataSet ds = builder.buildFromFile(config->dataSetFile, config->classColStart);
  for (int64_t i = 0; i < config->configTrees.size(); i++) {
    // Log starting test
    std::time_t start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    timeFile.open(timeFileName, std::ofstream::app);
    timeFile << "Starting test " << config->configTrees[i]->name << std::endl;
    timeFile << "Starting time: "
             << std::put_time(std::localtime(&start), "%Y-%m-%d_%H-%M-%S")
             << std::endl;
    timeFile.close();

    // Run test
    std::shared_ptr<DecisionTreeNode> tree = config->trees[i]->createTree(ds, config->configTrees[i]);
    ds.printTree(tree, outputFileName);
    tester.test(tree, ds, outputFileName);

    // Log finishing test
    std::time_t end = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::time_t elapsed = end - start;
    timeFile.open(timeFileName, std::ofstream::app);
    timeFile << "Finishing time: "
             << std::put_time(std::localtime(&end), "%Y-%m-%d_%H-%M-%S")
             << std::endl;
    timeFile << "Elapsed time in seconds "
             << end - start
             << std::endl;
    timeFile.close();
  }
}

