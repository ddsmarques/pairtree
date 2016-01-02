// This module implements the creation of a decision tree using a linear/integer programming algorithm.
// 
// Author: ddsmarques
//
#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "DecisionTreeNode.h"
#include "Tree.h"
#include "gurobi_c++.h"

class ConfigPine : public ConfigTree {
public:
  enum class SolverType { INTEGER, CONTINUOUS, CONTINUOUS_AFTER_ROOT };
  SolverType type;
};

class PineTree : public Tree {
public:
  PineTree();
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) override;
  void printBB();

private:
  std::shared_ptr<DecisionTreeNode> createBackBone(DataSet& ds, int64_t bbSize, ConfigPine::SolverType type);
  int64_t getBackBoneValue(DataSet& ds, int64_t col);
  void createVariables(DataSet& ds, int64_t bbSize);
  void createObjFunction(DataSet& ds, int64_t bbSize);
  void createConstraints(DataSet& ds, int64_t bbSize);
  void defineNodeLeaves(DataSet& ds, int64_t bbSize);
  void defineNodeLeavesInteger(DataSet& ds, int64_t bbSize);
  void defineNodeLeavesContinuous(DataSet& ds, int64_t bbSize);
  std::pair<double, int64_t> selectMaxPair(std::vector<std::pair<double, int64_t>>& v);
  std::shared_ptr<DecisionTreeNode> mountBackbone(DataSet& ds, int64_t bbSize);
  std::shared_ptr<DecisionTreeNode> mountBackboneLevel(DataSet& ds, int64_t level, int64_t bbSize);
  std::pair<int64_t, std::shared_ptr<DecisionTreeNode>> getNextNodeBB(std::shared_ptr<DecisionTreeNode> curr);
  template <typename T>
  void initMultidimension(std::vector<T>& v, std::vector<int64_t> dim);
  template <typename T>
  void initMultidimension(T& v, std::vector<int64_t> dim);

  std::shared_ptr<GRBEnv> env_;
  std::shared_ptr<GRBModel> model_;
  char varType_;
  int64_t totClasses_;
  int64_t totAttributes_;
  int64_t maxAttribSize_;
  std::vector<int64_t> bbValue_;
  std::vector<std::vector<GRBVar>> X_;
  std::vector<std::vector<GRBVar>> Z_;
  std::vector<std::vector<std::vector<GRBVar>>> W_;
  std::vector<std::vector<std::vector<std::vector<GRBVar>>>> V_;
  std::vector<std::vector<GRBVar>> L_;
  std::vector<GRBVar> Y_;
  std::vector<std::vector<int64_t>> nodeLeaves_;
  std::vector<int64_t> nodeAttrib_;
};

#include "PineTree-inl.h"
