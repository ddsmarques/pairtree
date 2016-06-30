#include "TrainReader.h"

#include "GreedyBBTree.h"
#include "GreedyDrawTree.h"
#include "GreedyTree.h"
#include "Logger.h"
#include "PairTree.h"
#include "PineTree.h"


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
    if (treeType.compare("pine") == 0) {
      std::shared_ptr<ConfigPine> pineConfig = std::make_shared<ConfigPine>();
      pineConfig->height = getVar<int>(tree, "height");
      pineConfig->name = getVar<std::string>(tree, "name");

      std::string solverType = getVar<std::string>(tree, "solver");
      if (solverType.compare("INTEGER") == 0) {
        pineConfig->type = ConfigPine::SolverType::INTEGER;
      }
      else if (solverType.compare("CONTINUOUS") == 0) {
        pineConfig->type = ConfigPine::SolverType::CONTINUOUS;
      }
      else if (solverType.compare("CONTINUOUS_AFTER_ROOT") == 0) {
        pineConfig->type = ConfigPine::SolverType::CONTINUOUS_AFTER_ROOT;
      }
      else {
        std::cout << "Error! Unknown solver type." << std::endl;
        return nullptr;
      }
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(pineConfig));

      std::shared_ptr<PineTree> pine = std::make_shared<PineTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(pine));

    }
    else if (treeType.compare("greedy") == 0) {
      std::shared_ptr<ConfigGreedy> gtConfig = std::make_shared<ConfigGreedy>();
      gtConfig->height = getVar<int>(tree, "height");
      gtConfig->name = getVar<std::string>(tree, "name");
      gtConfig->minLeaf = getVar<int>(tree, "minLeaf");
      gtConfig->percentiles = getVar<int>(tree, "percentiles");
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(gtConfig));

      std::shared_ptr<GreedyTree> greedy = std::make_shared<GreedyTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(greedy));

    }
    else if (treeType.compare("greedyBB") == 0) {
      std::shared_ptr<ConfigGreedyBB> gtConfig = std::make_shared<ConfigGreedyBB>();
      gtConfig->height = getVar<int>(tree, "height");
      gtConfig->name = getVar<std::string>(tree, "name");
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(gtConfig));

      std::shared_ptr<GreedyBBTree> greedyBB = std::make_shared<GreedyBBTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(greedyBB));
    }
    else if (treeType.compare("greedyDraw") == 0) {
      std::shared_ptr<ConfigGreedyDraw> drawConfig = std::make_shared<ConfigGreedyDraw>();
      drawConfig->height = getVar<int>(tree, "height");
      drawConfig->name = getVar<std::string>(tree, "name");
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
      pairConfig->maxBound = getVar<double>(tree, "maxBound");
      pairConfig->minLeaf = getVar<int>(tree, "minLeaf");
      pairConfig->useScore = getVar<bool>(tree, "useScore");
      luabridge::LuaRef alphas = tree["alphas"];
      int i = 0;
      while (!alphas[i].isNil()) {
        pairConfig->alphas.push_back(getVar<double>(alphas, i));
        i++;
      }
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(pairConfig));

      std::shared_ptr<PairTree> pairTree = std::make_shared<PairTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(pairTree));
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
