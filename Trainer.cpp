#include "Trainer.h"

#include "DataSet.h"
#include "DataSetBuilder.h"
#include "Tester.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <random>


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
  
  // Create model output file
  std::string outputFileName = folderName + "output.txt";
  std::ofstream outputFile;
  outputFile.open(outputFileName, std::ofstream::out);
  outputFile.close();

  // Create time log file
  std::string timeFileName = folderName + "time_log.txt";
  std::ofstream timeFile;
  timeFile.open(timeFileName, std::ofstream::out);
  timeFile << "Test name: " << config->name << std::endl;
  timeFile.close();

  createTrainTestDS(config);
  Tester tester;
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
    std::shared_ptr<DecisionTreeNode> tree = config->trees[i]->createTree(trainDS_, config->configTrees[i]);
    trainDS_.printTree(tree, outputFileName);
    tester.test(tree, testDS_, outputFileName);

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
#include <iostream>

void Trainer::createTrainTestDS(std::shared_ptr<ConfigTrain> config) {
  DataSetBuilder builder;
  trainDS_ = builder.buildFromFile(config->dataSetFile, config->classColStart);

  if (config->trainMode->type == ConfigTrainMode::trainType::TEST_SET) {
    testDS_ = builder.buildFromFile(config->trainMode->testFileName, config->classColStart);

  } else if (config->trainMode->type == ConfigTrainMode::trainType::TRAINING_SET) {
    testDS_ = trainDS_;

  } else if (config->trainMode->type == ConfigTrainMode::trainType::RANDOM_SPLIT) {
    testDS_.initAllAttributes(trainDS_);

    std::random_device rd;
    std::mt19937 gen(rd());
    int64_t trainSize = trainDS_.samples_.size() * config->trainMode->ratio;
    while (trainDS_.samples_.size() >= trainSize) {
      uint64_t next = std::uniform_int_distribution<uint64_t>{0, trainDS_.samples_.size()-1}(gen);
      
      auto it = trainDS_.samples_.begin();
      std::advance(it, next);
      testDS_.addSample(*it);
      trainDS_.eraseSample(it);
    }
  }
}
