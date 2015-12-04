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

class ConfigTrain {
public:
  std::string dataSetFile;
  std::string outputFolder;
  std::shared_ptr<ConfigTree> configTree;
  std::shared_ptr<Tree> tree;
};

class TrainReader {
public:
  std::shared_ptr<ConfigTrain> read(std::string fileName);

private:
  template <typename T>
  T getVar(luabridge::lua_State* L, std::string name);
  template <typename T>
  T getVar(luabridge::LuaRef& table, std::string name);
  luabridge::LuaRef getTable(luabridge::lua_State* L, std::string name);
};

#include "TrainReader-inl.h"
