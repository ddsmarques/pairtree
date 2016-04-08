#pragma once
#include "Tree.h"

#include <vector>


class ConfigPairTree : public ConfigTree {
public:
  double maxBound;
};

class PairTree : public Tree {
public:
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) override;

  struct SampleInfo {
    std::shared_ptr<Sample> ptr;
    int bestClass;
    double diff;
  };

private:
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int height, double maxBound);
  std::shared_ptr<DecisionTreeNode> createLeaf(DataSet& ds);
  long double getAttribScore(DataSet& ds, int64_t attribInx, std::vector<PairTree::SampleInfo>& samplesInfo);
  std::pair<long double, long double> getRandomScore(DataSet& ds, int64_t attribInx, std::vector<PairTree::SampleInfo>& samplesInfo);
  long double getProbBound(DataSet& ds, int64_t attribInx, std::vector<PairTree::SampleInfo>& samplesInfo, long double value);
  void initSampleInfo(DataSet& ds, std::vector<PairTree::SampleInfo>& samplesInfo);
};
