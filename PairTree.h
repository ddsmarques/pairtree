#pragma once
#include "Tree.h"

#include <functional>
#include <vector>


class ConfigPairTree : public ConfigTree {
public:
  double maxBound;
  int64_t minLeaf;
  bool useScore;
  bool useNominalBinary;
  bool useTBound;
  std::vector<long double> alphas;
  std::vector<int64_t> minSamples;
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
  struct AttribResult {
    long double score;
    int64_t separator;
    long double bound;
  };
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int height,
                                                  double maxBound,
                                                  int64_t minLeaf, bool useScore,
                                                  bool useNominalBinary,
                                                  bool useTBound);
  std::shared_ptr<DecisionTreeNode> createLeaf(DataSet& ds);
  void initSampleInfo(DataSet& ds, std::vector<PairTree::SampleInfo>& samplesInfo);

  AttribResult testAttribute(DataSet& ds, int64_t attribInx,
                             std::vector<PairTree::SampleInfo>& samplesInfo,
                             bool useNominalBinary, bool useTBound);
  AttribResult testNumeric(DataSet& ds, int64_t attribInx,
                           std::vector<PairTree::SampleInfo>& samplesInfo,
                           bool useTBound);
  AttribResult testNominal(DataSet& ds, int64_t attribInx,
                           std::vector<PairTree::SampleInfo>& samplesInfo,
                           bool useNominalBinary, bool useTBound);

  AttribScoreResult calcNominalScore(DataSet& ds, int64_t attribInx,
                                     std::function<int64_t(int64_t)> valueBox,
                                     int64_t attribSize,
                                     std::vector<PairTree::SampleInfo>& samplesInfo);

  long double getAttribBound(DataSet& ds, AttribScoreResult& attribResult, int64_t attribInx,
                             std::vector<PairTree::SampleInfo>& samplesInfo,
                             bool useTBound);
  std::pair<long double, long double> getRandomScore(std::vector<PairTree::SampleInfo>& samplesInfo,
                                                     std::vector<double>& distrib);
  long double getProbBound(DataSet& ds, int64_t attribInx,
                           std::vector<PairTree::SampleInfo>& samplesInfo,
                           long double t, bool useTBound);
  long double calcTSqVar(DataSet& ds);
  long double calcSum(const std::vector<long double>& v, int64_t i, int64_t j);
  long double calcMatchingSums(const std::vector<long double>& a, const std::vector<long double>& b);
  std::pair<long double, long double> calcXstarSumsq(int64_t attribInx,
                                                     std::vector<PairTree::SampleInfo>& samplesInfo);
  long double applyBound(long double t, long double xstar, long double sumSqBounds);
  
};
