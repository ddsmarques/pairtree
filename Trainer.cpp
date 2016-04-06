#include "Trainer.h"

#include "DataSet.h"
#include "DataSetBuilder.h"
#include "Logger.h"
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

  // Create time log file
  Logger::setOutput(outputFolder_ + "log.txt");
  Logger::log() << "Test name: " << config->name;

  // Copy input file
  cmd = "copy " + fileName + " " + outputFolder_;
  system(cmd.c_str());

  // Create summary file
  std::string summaryFileName = outputFolder_ + "summary.txt";
  std::ofstream summaryFile;
  summaryFile.open(summaryFileName, std::ofstream::out);
  summaryFile << "Summary file" << std::endl;
  summaryFile.close();

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
    testDS.initAllAttributes(trainDS);
  } else if (config->trainMode->type == ConfigTrainMode::trainType::SPLIT) {
    trainDS = builder.buildFromFile(config->dataSetFile, config->classColStart);
  }

  for (int64_t i = 0; i < config->configTrees.size(); i++) {
    double score = 0;
    if (config->trainMode->type == ConfigTrainMode::trainType::RANDOM_SPLIT) {
      for (int fold = 0; fold < config->trainMode->folds; fold++) {
        getRandomSplit(trainDS, testDS, config->trainMode->ratio);
        score += runTree(config, i, trainDS, testDS);
      }
      score = score / config->trainMode->folds;
    } else if (config->trainMode->type == ConfigTrainMode::trainType::SPLIT) {
      for (int fold = 0; fold < config->trainMode->folds; fold++) {
        DataSet currTrain, currTest;
        currTrain.initAllAttributes(trainDS);
        currTest.initAllAttributes(trainDS);
        getSplit(trainDS, currTrain, currTest, config->dataSetFile, fold+1);
        score += runTree(config, i, currTrain, currTest);
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

  Logger::closeOutput();
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


void Trainer::getSplit(DataSet& originalDS, DataSet& currTrain,
                       DataSet& currTest, std::string fileName,
                       int fold) {
  std::string trainName = fileName + "_train_" + std::to_string(fold);
  std::string testName = fileName + "_test_" + std::to_string(fold);
  loadSplit(originalDS, currTrain, trainName);
  loadSplit(originalDS, currTest, testName);
}


void Trainer::loadSplit(DataSet& originalDS, DataSet& current, std::string fileName) {
  std::ifstream file(fileName);
  int64_t totSamples = 0;
  file >> totSamples;
  while (totSamples > 0) {
    int64_t inx;
    file >> inx;
    auto it = originalDS.samples_.begin();
    std::advance(it, inx);
    current.samples_.push_back(*it);

    totSamples--;
  }
}


double Trainer::runTree(std::shared_ptr<ConfigTrain>& config, int treeInx,
                        DataSet& trainDS, DataSet& testDS) {
  // Log starting test
  auto start = std::chrono::system_clock::now();
  Logger::log() << "Starting test " << config->configTrees[treeInx]->name;

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
  Logger::log() << "Finished test " << config->configTrees[treeInx]->name;
  int64_t countSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count();
  Logger::log() << "Elapsed time in seconds " << std::to_string(countSeconds);

  return score;
}

