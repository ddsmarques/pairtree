#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "DecisionTreeNode.h"
#include "Tree.h"

#include <vector>

class ConfigGreedy : public ConfigTree {
public:
  int64_t minLeaf;
  int64_t percentiles;
  double minGain;
  bool useNominalBinary;
  std::vector<long double> alphas;
  std::vector<int64_t> minSamples;
};

class GreedyTree : public Tree {
public:
  GreedyTree();
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) override;

private:
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height,
                                                  int64_t minLeaf, int64_t percentiles,
                                                  double minGain, bool useNominalBinary);
  bool isAllSameClass(DataSet& ds);
  std::pair<long double, int64_t> getAttribScore(DataSet& ds, int64_t attribInx,
                                                 int64_t percentiles, bool useNominalBinary);
  std::pair<long double, int64_t> getNominalScore(DataSet& ds, int64_t attribInx, bool useNominalBinary);
  std::pair<long double, int64_t> getOrderedScore(DataSet& ds, int64_t attribInx,
                                                  int64_t percentiles);
  std::shared_ptr<DecisionTreeNode> createLeaf(DataSet& ds);
  long double calcGain(DataSet& ds, long double score);
};
