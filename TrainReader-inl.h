
template <typename T>
T TrainReader::getVar(luabridge::lua_State* L, std::string name) {
  auto ans = luabridge::getGlobal(L, name.c_str());
  ErrorUtils::enforce(!ans.isNil(), "Can't find variable " + name);
  return ans.cast<T>();
}

template <typename T>
T TrainReader::getVar(luabridge::LuaRef& table, std::string name) {
  auto ans = table[name.c_str()];
  ErrorUtils::enforce(!ans.isNil(), "Can't find variable " + name);
  return ans.cast<T>();
}

template <typename T>
T TrainReader::getVar(luabridge::LuaRef& table, int index) {
  auto ans = table[index];
  ErrorUtils::enforce(!ans.isNil(),
                      "Can't find variable at index " + std::to_string(index));
  return ans.cast<T>();
}
