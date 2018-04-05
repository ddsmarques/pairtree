// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#pragma once
#include "ConfigTree.h"
#include "Tree.h"

#include <LuaBridge.h>
extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

#include <string>
#include <memory>
#include <vector>

class ConfigTrainMode {
public:
  enum class trainType {RANDOM_SPLIT, SPLIT, TEST_SET, TRAINING_SET};

  trainType type;
  std::string testFileName;
  double ratio;
  int folds;
};

class ConfigTrain {
public:
  std::string dataSetFile;
  int64_t classColStart;
  std::string outputFolder;
  std::string name;
  std::vector<std::shared_ptr<ConfigTree>> configTrees;
  std::vector<std::shared_ptr<Tree>> trees;
  std::shared_ptr<ConfigTrainMode> trainMode;
};

class TrainReader {
public:
  std::shared_ptr<ConfigTrain> read(std::string fileName);

private:
  template <typename T>
  T getVar(luabridge::lua_State* L, std::string name);
  template <typename T>
  T getVar(luabridge::LuaRef& table, std::string name);
  template <typename T>
  T getVar(luabridge::LuaRef& table, int index);
  luabridge::LuaRef getTable(luabridge::lua_State* L, std::string name);
};

#include "TrainReader-inl.h"
