// This module implements the creation of a decision tree using a linear/integer programming algorithm.
// 
// Author: ddsmarques
//
#pragma once
#include "DataSet.h"
#include "DecisionTreeNode.h"
#include "gurobi_c++.h"

class PineTree {
public:
  enum class SolverType {INTEGER, CONTINUOUS};
  PineTree();
  void createBackBone(DataSet& ds, int64_t bbSize, SolverType type);
  std::shared_ptr<DecisionTreeNode> mountBackbone(DataSet& ds);

private:
  int64_t getBackBoneValue(DataSet& ds, int64_t col);
  void printBB();
  void createVariables(DataSet& ds);
  void createObjFunction(DataSet& ds);
  void createConstraints(DataSet& ds);
  void defineNodeLeaves(DataSet& ds);
  void defineNodeLeavesInteger(DataSet& ds);
  void defineNodeLeavesContinuous(DataSet& ds);
  std::pair<double, int64_t> selectMaxPair(std::vector<std::pair<double, int64_t>>& v);
  std::shared_ptr<DecisionTreeNode> mountBackboneLevel(DataSet& ds, int64_t level);
  template <typename T>
  void initMultidimension(std::vector<T>& v, std::vector<int64_t> dim);
  template <typename T>
  void initMultidimension(T& v, std::vector<int64_t> dim);

  GRBEnv env_;
  GRBModel model_;
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
