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
  std::string boundOption;
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
  enum BoundType { DIFF_BOUND, T_BOUND, VAR_BOUND };
  struct BoundConstants {
    long double xstar;
    long double sumDSq;
    long double S;
    long double b;
    long double TSq;
  };
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
                                                  BoundType boundType);
  std::shared_ptr<DecisionTreeNode> createLeaf(DataSet& ds);
  bool isAllSameClass(DataSet& ds);
  void initSampleInfo(DataSet& ds, std::vector<PairTree::SampleInfo>& samplesInfo);

  AttribResult testAttribute(DataSet& ds, int64_t attribInx,
                             std::vector<PairTree::SampleInfo>& samplesInfo,
                             bool useNominalBinary, BoundType boundType);
  AttribResult testNumeric(DataSet& ds, int64_t attribInx,
                           std::vector<PairTree::SampleInfo>& samplesInfo,
                           BoundType boundType);
  AttribResult testNominal(DataSet& ds, int64_t attribInx,
                           std::vector<PairTree::SampleInfo>& samplesInfo,
                           bool useNominalBinary, BoundType boundType);

  AttribScoreResult calcNominalScore(DataSet& ds, int64_t attribInx,
                                     std::function<int64_t(int64_t)> valueBox,
                                     int64_t attribSize,
                                     std::vector<PairTree::SampleInfo>& samplesInfo);

  std::pair<long double, long double> getRandomScore(std::vector<PairTree::SampleInfo>& samplesInfo,
                                                     std::vector<double>& distrib);

  long double getAttribBound(DataSet& ds, AttribScoreResult& attribResult, int64_t attribInx,
                             std::vector<PairTree::SampleInfo>& samplesInfo,
                             BoundType boundType);
  long double applyBound(long double t, BoundConstants constants, BoundType boundType);
  BoundConstants calcConstants(DataSet& ds, int64_t attribInx,
                               std::vector<PairTree::SampleInfo>& samplesInfo,
                               BoundType boundType);

  long double calcConstXstar(DataSet& ds);
  long double calcConstSumDSq(int64_t attribInx,
                              std::vector<PairTree::SampleInfo>& samplesInfo);
  long double calcConstTSq(DataSet& ds);
  long double calcConstS(DataSet& ds, int64_t attribInx, std::vector<PairTree::SampleInfo>& samplesInfo);
  long double calcConstb(DataSet& ds, int64_t attribInx);
  long double calcSum(const std::vector<long double>& v, int64_t i, int64_t j);
  long double calcMatchingSums(const std::vector<long double>& a, const std::vector<long double>& b);
  long double calcVarSums(const std::vector<long double>& a, const std::vector<long double>& b);
  void createTwoDiffs(DataSet& ds, std::vector<long double>& s0, std::vector<long double>& s1);
  long double calcMaxD(DataSet& ds);
  long double calcSplitProb(DataSet& ds, int64_t attribInx);
  
  
};
