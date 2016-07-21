#include "TrainReader.h"

#include "AodhaTree.h"
#include "GreedyBBTree.h"
#include "GreedyDrawTree.h"
#include "GreedyTree.h"
#include "Logger.h"
#include "PairTree.h"

#include <iostream>


std::shared_ptr<ConfigTrain> TrainReader::read(std::string fileName) {
  std::shared_ptr<ConfigTrain> config = std::make_shared<ConfigTrain>();
  luabridge::lua_State* L = luabridge::luaL_newstate();
  ErrorUtils::enforce(luaL_dofile(L, fileName.c_str()) == 0, "Error opening config file " + fileName);
  luabridge::luaL_openlibs(L);
  lua_pcall(L, 0, 0, 0);
  config->outputFolder = getVar<std::string>(L, "output");
  config->name = getVar<std::string>(L, "name");

  auto dataset = getTable(L, "dataset");
  if (!dataset.isNil()) {
    config->dataSetFile = getVar<std::string>(dataset, "filename");
    config->classColStart = getVar<int>(dataset, "classColStart");
  } else {
    std::cout << "Invalid dataset table." << std::endl;
    return nullptr;
  }
  auto trainMode = getTable(L, "trainMode");
  if (!trainMode.isNil()) {
    std::string auxType = getVar<std::string>(trainMode, "trainType");
    if (auxType.compare("randomsplit") == 0) {
      config->trainMode = std::make_shared<ConfigTrainMode>();
      config->trainMode->type = ConfigTrainMode::trainType::RANDOM_SPLIT;
      config->trainMode->ratio = getVar<double>(trainMode, "ratio");
      config->trainMode->folds = getVar<int>(trainMode, "folds");
    } else if (auxType.compare("split") == 0) {
      config->trainMode = std::make_shared<ConfigTrainMode>();
      config->trainMode->type = ConfigTrainMode::trainType::SPLIT;
      config->trainMode->folds = getVar<int>(trainMode, "folds");
    } else if (auxType.compare("testset") == 0) {
      config->trainMode = std::make_shared<ConfigTrainMode>();
      config->trainMode->type = ConfigTrainMode::trainType::TEST_SET;
      config->trainMode->testFileName = getVar<std::string>(trainMode, "filename");
    } else if (auxType.compare("trainingset") == 0) {
      config->trainMode = std::make_shared<ConfigTrainMode>();
      config->trainMode->type = ConfigTrainMode::trainType::TRAINING_SET;
    } else {
      std::cout << "Invalid trainType value." << std::endl;
      return nullptr;
    }
  } else {
    std::cout << "Invalid trainMode table." << std::endl;
    return nullptr;
  }

  auto trees = getTable(L, "trees");
  int count = 0;
  while (!trees[count].isNil()) {
    luabridge::LuaRef tree = trees[count];
    std::string treeType = getVar<std::string>(tree, "treeType");
    if (treeType.compare("greedy") == 0) {
      std::shared_ptr<ConfigGreedy> gtConfig = std::make_shared<ConfigGreedy>();
      gtConfig->height = getVar<int>(tree, "height");
      gtConfig->name = getVar<std::string>(tree, "name");
      gtConfig->typeName = treeType;
      gtConfig->minLeaf = getVar<int>(tree, "minLeaf");
      gtConfig->percentiles = getVar<int>(tree, "percentiles");
      gtConfig->minGain = getVar<double>(tree, "minGain");
      luabridge::LuaRef minSamples = tree["minSamples"];
      int i = 0;
      while (!minSamples[i].isNil()) {
        gtConfig->minSamples.push_back(getVar<int>(minSamples, i));
        i++;
      }
      luabridge::LuaRef alphas = tree["alphas"];
      i = 0;
      while (!alphas[i].isNil()) {
        gtConfig->alphas.push_back(getVar<double>(alphas, i));
        i++;
      }
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(gtConfig));

      std::shared_ptr<GreedyTree> greedy = std::make_shared<GreedyTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(greedy));

    }
    else if (treeType.compare("greedyBB") == 0) {
      std::shared_ptr<ConfigGreedyBB> gtConfig = std::make_shared<ConfigGreedyBB>();
      gtConfig->height = getVar<int>(tree, "height");
      gtConfig->name = getVar<std::string>(tree, "name");
      gtConfig->typeName = treeType;
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(gtConfig));

      std::shared_ptr<GreedyBBTree> greedyBB = std::make_shared<GreedyBBTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(greedyBB));
    }
    else if (treeType.compare("greedyDraw") == 0) {
      std::shared_ptr<ConfigGreedyDraw> drawConfig = std::make_shared<ConfigGreedyDraw>();
      drawConfig->height = getVar<int>(tree, "height");
      drawConfig->name = getVar<std::string>(tree, "name");
      drawConfig->typeName = treeType;
      drawConfig->totDraws = getVar<int>(tree, "totDraws");
      drawConfig->minLeaf = getVar<int>(tree, "minLeaf");
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(drawConfig));

      std::shared_ptr<GreedyDrawTree> greedyDraw = std::make_shared<GreedyDrawTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(greedyDraw));
    }
    else if (treeType.compare("pair") == 0) {
      std::shared_ptr<ConfigPairTree> pairConfig = std::make_shared<ConfigPairTree>();
      pairConfig->height = getVar<int>(tree, "height");
      pairConfig->name = getVar<std::string>(tree, "name");
      pairConfig->typeName = treeType;
      pairConfig->maxBound = getVar<double>(tree, "maxBound");
      pairConfig->minLeaf = getVar<int>(tree, "minLeaf");
      pairConfig->useScore = getVar<bool>(tree, "useScore");
      pairConfig->useNominalBinary = getVar<bool>(tree, "useNominalBinary");
      pairConfig->boundOption = getVar<int>(tree, "boundOption");
      luabridge::LuaRef minSamples = tree["minSamples"];
      int i = 0;
      while (!minSamples[i].isNil()) {
        pairConfig->minSamples.push_back(getVar<int>(minSamples, i));
        i++;
      }
      luabridge::LuaRef alphas = tree["alphas"];
      i = 0;
      while (!alphas[i].isNil()) {
        pairConfig->alphas.push_back(getVar<double>(alphas, i));
        i++;
      }
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(pairConfig));

      std::shared_ptr<PairTree> pairTree = std::make_shared<PairTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(pairTree));
    }
    else if (treeType.compare("aodha") == 0) {
      std::shared_ptr<ConfigAodha> aodhaConfig = std::make_shared<ConfigAodha>();
      aodhaConfig->height = getVar<int>(tree, "height");
      aodhaConfig->name = getVar<std::string>(tree, "name");
      aodhaConfig->typeName = treeType;
      aodhaConfig->minLeaf = getVar<int>(tree, "minLeaf");
      aodhaConfig->minGain = getVar<double>(tree, "minGain");
      luabridge::LuaRef minSamples = tree["minSamples"];
      int i = 0;
      while (!minSamples[i].isNil()) {
        aodhaConfig->minSamples.push_back(getVar<int>(minSamples, i));
        i++;
      }
      luabridge::LuaRef alphas = tree["alphas"];
      i = 0;
      while (!alphas[i].isNil()) {
        aodhaConfig->alphas.push_back(getVar<double>(alphas, i));
        i++;
      }
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(aodhaConfig));

      std::shared_ptr<AodhaTree> aodhaTree = std::make_shared<AodhaTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(aodhaTree));
    }
    else {
      std::cout << "Error! Unknown tree type." << std::endl;
      return nullptr;
    }
    count++;
  }
  return config;
}

luabridge::LuaRef TrainReader::getTable(luabridge::lua_State* L, std::string name) {
  luabridge::LuaRef table = luabridge::getGlobal(L, name.c_str());
  ErrorUtils::enforce(!table.isNil(), "Can't find table " + name);
  return table;
}
