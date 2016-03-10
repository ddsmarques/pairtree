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
  outputFolder_ = config->outputFolder + config->name + "_" + buffer.str() + "\\";
  std::string cmd = "mkdir " + outputFolder_;
  system(cmd.c_str());

  // Create summary file
  std::string summaryFileName = outputFolder_ + "summary.txt";
  std::ofstream summaryFile;
  summaryFile.open(summaryFileName, std::ofstream::out);
  summaryFile << "Summary file" << std::endl;
  summaryFile.close();

  // Create time log file
  timeFileName_ = outputFolder_ + "time_log.txt";
  std::ofstream timeFile;
  timeFile.open(timeFileName_, std::ofstream::out);
  timeFile << "Test name: " << config->name << std::endl;
  timeFile.close();

  DataSetBuilder builder;
  DataSet trainDS;
  DataSet testDS;
  if (config->trainMode->type == ConfigTrainMode::trainType::TEST_SET) {
    trainDS = builder.buildFromFile(config->dataSetFile, config->classColStart);
    testDS = builder.buildFromFile(config->trainMode->testFileName, config->classColStart);
  }
  else if (config->trainMode->type == ConfigTrainMode::trainType::TRAINING_SET) {
    trainDS = builder.buildFromFile(config->dataSetFile, config->classColStart);
    testDS = trainDS;
  }
  else if (config->trainMode->type == ConfigTrainMode::trainType::RANDOM_SPLIT) {
    trainDS = builder.buildFromFile(config->dataSetFile, config->classColStart);
    testDS;
    testDS.initAllAttributes(trainDS);
  }

  for (int64_t i = 0; i < config->configTrees.size(); i++) {
    double score = 0;
    if (config->trainMode->type == ConfigTrainMode::trainType::RANDOM_SPLIT) {
      for (int fold = 0; fold < config->trainMode->folds; fold++) {
        getRandomSplit(trainDS, testDS, config->trainMode->ratio);
        score += runTree(config, i, trainDS, testDS);
      }
      score = score / config->trainMode->folds;
    } else {
      score = runTree(config, i, trainDS, testDS);
    }

    // Log score
    summaryFile.open(summaryFileName, std::ofstream::app);
    summaryFile << config->configTrees[i]->name << " " << score << std::endl;
    summaryFile.close();
  }
}


void Trainer::getRandomSplit(DataSet& trainDS, DataSet& testDS, double ratio) {
  // Puts all testDS samples to trainDS
  while (testDS.samples_.size() > 0) {
    trainDS.addSample(*testDS.samples_.begin());
    testDS.eraseSample(testDS.samples_.begin());
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  int64_t trainSize = trainDS.samples_.size() * ratio;
  while (trainDS.samples_.size() >= trainSize) {
    uint64_t next = std::uniform_int_distribution<uint64_t>{ 0, trainDS.samples_.size() - 1 }(gen);

    auto it = trainDS.samples_.begin();
    std::advance(it, next);
    testDS.addSample(*it);
    trainDS.eraseSample(it);
  }
}

double Trainer::runTree(std::shared_ptr<ConfigTrain>& config, int treeInx,
                        DataSet& trainDS, DataSet& testDS) {
  // Log starting test
  std::ofstream timeFile;
  std::time_t start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  timeFile.open(timeFileName_, std::ofstream::app);
  timeFile << "Starting test " << config->configTrees[treeInx]->name << std::endl;
  timeFile << "Starting time: "
    << std::put_time(std::localtime(&start), "%Y-%m-%d_%H-%M-%S")
    << std::endl;
  timeFile.close();

  // Create model output file
  std::string outputFileName = outputFolder_ + "outputTree_" + config->configTrees[treeInx]->name + ".txt";
  std::ofstream outputFile;
  outputFile.open(outputFileName, std::ofstream::app);
  outputFile.close();

  // Run test
  std::shared_ptr<DecisionTreeNode> tree = config->trees[treeInx]->createTree(trainDS, config->configTrees[treeInx]);
  trainDS.printTree(tree, outputFileName);
  Tester tester;
  double score = tester.test(tree, testDS, outputFileName);

  // Log finishing test
  std::time_t end = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::time_t elapsed = end - start;
  timeFile.open(timeFileName_, std::ofstream::app);
  timeFile << "Finishing time: "
    << std::put_time(std::localtime(&end), "%Y-%m-%d_%H-%M-%S")
    << std::endl;
  timeFile << "Elapsed time in seconds "
    << end - start
    << std::endl;
  timeFile.close();

  return score;
}

