#include "TrainReader.h"

#include "GreedyTree.h"
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
  auto trees = getTable(L, "trees");
  int count = 0;
  while (!trees[count].isNil()) {
    luabridge::LuaRef tree = trees[count];
    std::string treeType = getVar<std::string>(tree, "treeType");
    if (treeType.compare("pine") == 0) {
      std::shared_ptr<ConfigPine> pineConfig = std::make_shared<ConfigPine>();
      pineConfig->height = getVar<int>(tree, "height");

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
      config->configTrees.push_back(std::static_pointer_cast<ConfigTree>(gtConfig));

      std::shared_ptr<GreedyTree> greedy = std::make_shared<GreedyTree>();
      config->trees.push_back(std::dynamic_pointer_cast<Tree>(greedy));

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
