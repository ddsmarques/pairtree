#pragma once
#include "Tree.h"

#include <vector>


class ConfigPairTree : public ConfigTree {
public:
  double maxBound;
  int64_t minLeaf;
  std::vector<long double> alphas;
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
  struct AttribScoreResult {
    long double score;
    int64_t separator;
    std::vector<double> distrib;
  };
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int height,
                                                  double maxBound, int64_t minLeaf);
  std::shared_ptr<DecisionTreeNode> createLeaf(DataSet& ds);
  std::pair<long double, int64_t> testAttribute(DataSet& ds, int64_t attribInx, std::vector<PairTree::SampleInfo>& samplesInfo);
  AttribScoreResult getNominalAttribScore(DataSet& ds, int64_t attribInx,
                                          std::vector<PairTree::SampleInfo>& samplesInfo);
  AttribScoreResult getOrderedAttribScore(DataSet& ds, int64_t attribInx,
                                          std::vector<PairTree::SampleInfo>& samplesInfo);
  long double getAttribBound(AttribScoreResult& attribResult, int64_t attribInx,
                             std::vector<PairTree::SampleInfo>& samplesInfo);
  std::pair<long double, long double> getRandomScore(std::vector<PairTree::SampleInfo>& samplesInfo,
                                                     std::vector<double>& distrib);
  long double getProbBound(int64_t attribInx, int64_t attribSize,
                          std::vector<PairTree::SampleInfo>& samplesInfo, long double value,
                          int64_t separator);
  int64_t getBinBox(int64_t attribValue, int64_t separator);
  void initSampleInfo(DataSet& ds, std::vector<PairTree::SampleInfo>& samplesInfo);
};
