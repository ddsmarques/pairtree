#include "Trainer.h"

#include "CompareUtils.h"
#include "DataSet.h"
#include "DataSetBuilder.h"
#include "GreedyTree.h"
#include "Logger.h"
#include "PairTree.h"
#include "ExtrasTreeNode.h"
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
    long double score = 0;
    long double savings = 0;
    double size = 0;
    int64_t totSeconds = 0;
    if (config->trainMode->type == ConfigTrainMode::trainType::RANDOM_SPLIT) {
      auto runResult = execRandomSplit(config, trainDS, testDS, i);
      score = runResult.score;
      savings = runResult.savings;
      size = runResult.size;
      totSeconds += runResult.seconds;
    } else if (config->trainMode->type == ConfigTrainMode::trainType::SPLIT) {
      auto runResult = execSplit(config, trainDS, i);
      score = runResult.score;
      savings = runResult.savings;
      size = runResult.size;
      totSeconds += runResult.seconds;
    } else {
      auto result = runTree(config, i, trainDS, testDS);
      score = result.score;
      savings = result.savings;
      size = result.size;
      totSeconds = result.seconds;
    }
    Logger::log() << "Total elapsed time (s): " << totSeconds;

    // Log score
    summaryFile.open(summaryFileName, std::ofstream::app);
    summaryFile << config->configTrees[i]->name << std::endl
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << score << std::endl
                << savings << std::endl
                << size << std::endl;
    summaryFile.close();
  }

  Logger::closeOutput();
}


Trainer::TreeResult Trainer::execRandomSplit(std::shared_ptr<ConfigTrain> config,
                                             DataSet& trainDS, DataSet& testDS,
                                             int treeInx) {
  TreeResult runResult;
  runResult.score = 0;
  runResult.savings = 0;
  runResult.size = 0;
  runResult.seconds = 0;

  for (int fold = 0; fold < config->trainMode->folds; fold++) {
    getRandomSplit(trainDS, testDS, config->trainMode->ratio);
    auto result = runTree(config, treeInx, trainDS, testDS);
    runResult.score += result.score;
    runResult.savings += result.savings;
    runResult.size += result.size;
    runResult.seconds += result.seconds;
  }

  return runResult;
}


Trainer::TreeResult Trainer::execSplit(std::shared_ptr<ConfigTrain> config,
                                       DataSet& trainDS, int treeInx) {
  TreeResult runResult;
  runResult.score = 0;
  runResult.savings = 0;
  runResult.size = 0;
  runResult.seconds = 0;

  for (int fold = 0; fold < config->trainMode->folds; fold++) {
    DataSet currTrain, currTest;
    currTrain.initAllAttributes(trainDS);
    currTest.initAllAttributes(trainDS);
    getSplit(trainDS, currTrain, currTest, config->dataSetFile, fold);
    auto foldResult = runTree(config, treeInx, currTrain, currTest);

    if (foldResult.alphaXsamples.size() > 0) {
      // Update sum of all alphaXsamples matrix
      if (runResult.alphaXsamples.size() == 0) {
        runResult.alphaXsamples = foldResult.alphaXsamples;
      } else {
        if (runResult.alphaXsamples.size() == foldResult.alphaXsamples.size()
            && runResult.alphaXsamples[0].size() == foldResult.alphaXsamples[0].size()) {
          for (int i = 0; i < foldResult.alphaXsamples.size(); i++) {
            for (int j = 0; j < foldResult.alphaXsamples[i].size(); j++) {
              runResult.alphaXsamples[i][j] += foldResult.alphaXsamples[i][j];
            }
          }
        } else {
          Logger::log() << "Error: Folds with different sizes of the alphaXsamples matrix.";
        }
      }

      // Saves this fold alphaXsamples matrix
      std::string foldFileName = outputFolder_ + "alphaXsamples_fold" + std::to_string(fold) + ".csv";
      std::ofstream ofs(foldFileName, std::ofstream::out);
      for (int i = 0; i < foldResult.alphaXsamples.size(); i++) {
        for (int j = 0; j < foldResult.alphaXsamples[i].size(); j++) {
          if (j != 0) ofs << ",";
          ofs << foldResult.alphaXsamples[i][j];
        }
        ofs << std::endl;
      }
      ofs.close();
    }

    runResult.score += foldResult.score;
    runResult.savings += foldResult.savings;
    runResult.size += foldResult.size;
    runResult.seconds += foldResult.seconds;
  }

  runResult.score /= config->trainMode->folds;
  runResult.savings /= config->trainMode->folds;
  runResult.size /= config->trainMode->folds;
  for (int i = 0; i < runResult.alphaXsamples.size(); i++) {
    for (int j = 0; j < runResult.alphaXsamples[i].size(); j++) {
      runResult.alphaXsamples[i][j] /= config->trainMode->folds;
    }
  }

  // Saves this fold alphaXsamples matrix
  std::string allFoldsFileName = outputFolder_ + "alphaXsamples_AllFolds.csv";
  std::ofstream ofs(allFoldsFileName, std::ofstream::out);
  for (int i = 0; i < runResult.alphaXsamples.size(); i++) {
    for (int j = 0; j < runResult.alphaXsamples[i].size(); j++) {
      if (j != 0) ofs << ",";
      ofs << runResult.alphaXsamples[i][j];
    }
    ofs << std::endl;
  }
  ofs.close();

  return runResult;
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
  std::string trainName = fileName.substr(0, fileName.size() - 4) + "_train_" + std::to_string(fold) + ".txt";
  std::string testName = fileName.substr(0, fileName.size() - 4) + "_val_" + std::to_string(fold) + ".txt";
  loadSplit(originalDS, currTrain, trainName);
  loadSplit(originalDS, currTest, testName);
}


void Trainer::loadSplit(DataSet& originalDS, DataSet& current, std::string fileName) {
  std::ifstream file(fileName);
  int64_t totSamples = 0;
  file >> totSamples;
  // Add all indexes to a vector and sorts it
  std::vector<int64_t> indexes(totSamples);
  for (int64_t i = 0; i < totSamples; i++) {
    file >> indexes[i];
  }
  std::sort(indexes.begin(), indexes.end());

  // Creates dataset in O(N)
  auto it = originalDS.samples_.begin();
  for (int64_t i = 0; i < totSamples; i++) {
    if (i == 0) {
      std::advance(it, indexes[i]);
    } else {
      std::advance(it, indexes[i] - indexes[i-1]);
    }
    current.samples_.push_back(*it);
  }
}


Trainer::TreeResult Trainer::runTree(std::shared_ptr<ConfigTrain>& config, int treeInx,
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
  Tester tester;
  TreeResult treeResult;
  Tester::TestResults testResult;
  if (std::dynamic_pointer_cast<PairTree>(config->trees[treeInx]) != nullptr
      || std::dynamic_pointer_cast<GreedyTree>(config->trees[treeInx]) != nullptr) {
    treeResult = runAlphaSamplesTrees(config, treeInx, trainDS, testDS);
    testResult.savings = treeResult.savings;
    testResult.score = treeResult.score;
    testResult.size = treeResult.size;
  } else {
    std::shared_ptr<DecisionTreeNode> tree = config->trees[treeInx]->createTree(trainDS,
                                                config->configTrees[treeInx]);
    trainDS.printTree(tree, outputFileName);
    testResult = tester.test(tree, testDS);
    treeResult.savings = testResult.savings;
    treeResult.score = testResult.score;
    treeResult.size = testResult.size;
  }
  tester.saveResult(testResult, outputFileName);

  // Log finishing test
  Logger::log() << "Finished test " << config->configTrees[treeInx]->name;
  int64_t countSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count();
  Logger::log() << "Elapsed time in seconds " << countSeconds;

  treeResult.seconds = countSeconds;

  return treeResult;
}


Trainer::TreeResult Trainer::runAlphaSamplesTrees(std::shared_ptr<ConfigTrain>& config,
                                                  int treeInx,
                                                  DataSet& trainDS, DataSet& testDS) {
  std::vector<long double> alphas;
  std::vector<int64_t> minSamples;

  if (std::static_pointer_cast<PairTree>(config->trees[treeInx]) != nullptr) {
    alphas = std::static_pointer_cast<ConfigPairTree>(config->configTrees[treeInx])->alphas;
    minSamples = std::static_pointer_cast<ConfigPairTree>(config->configTrees[treeInx])->minSamples;
  } else if (std::static_pointer_cast<GreedyTree>(config->trees[treeInx]) != nullptr) {
    alphas = std::static_pointer_cast<ConfigGreedy>(config->configTrees[treeInx])->alphas;
    minSamples = std::static_pointer_cast<ConfigGreedy>(config->configTrees[treeInx])->minSamples;
  } else {
    Logger::log() << "Error training trees using multiple alphas and samples.";
    TreeResult ans;
    ans.score = -1;
    ans.savings = -1;
    ans.size = -1;
    return ans;
  }

  TreeResult treeResult;
  treeResult.score = 0;
  treeResult.savings = 0;
  treeResult.size = 0;
  treeResult.alphaXsamples = std::vector<std::vector<long double>>(alphas.size(),
                                                                   std::vector<long double>(minSamples.size()));
  Tester tester;
  std::shared_ptr<ExtrasTreeNode> fullTree = std::static_pointer_cast<ExtrasTreeNode>(
                                              config->trees[treeInx]->createTree(trainDS, config->configTrees[treeInx]));
  for (int i = 0; i < alphas.size(); i++) {
    auto alpha = alphas[i];
    for (int j = 0; j < minSamples.size(); j++) {
      auto samples = minSamples[j];
      auto alphaSampleTree = fullTree->getTree(alpha, samples);
      auto alphaSampleResult = tester.test(alphaSampleTree, testDS);

      treeResult.score += alphaSampleResult.score;
      treeResult.savings += alphaSampleResult.savings;
      treeResult.size += alphaSampleResult.size;
      treeResult.alphaXsamples[i][j] = alphaSampleResult.savings;

      // Print tree to this alpha X samples file
      std::string alphaSampleFileName = outputFolder_ + "outputTree_" + config->name
        + "_alphaXsample_" + std::to_string(alpha) + "X" + std::to_string(samples) + ".txt";
      trainDS.printTree(alphaSampleTree, alphaSampleFileName);
      tester.saveResult(alphaSampleResult, alphaSampleFileName);
    }
  }

  treeResult.score /= alphas.size() * minSamples.size();
  treeResult.savings /= alphas.size() * minSamples.size();
  treeResult.size /= alphas.size() * minSamples.size();

  return treeResult;
}
