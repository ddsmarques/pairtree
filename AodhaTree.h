// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "DecisionTreeNode.h"
#include "Tree.h"

#include <vector>

class ConfigAodha : public ConfigTree {
public:
  int64_t minLeaf;
  double minGain;
  bool useNominalBinary;
  std::vector<long double> alphas;
  std::vector<int64_t> minSamples;
};

class AodhaTree : public Tree {
public:
  AodhaTree();
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) override;

private:
  struct AttribResult {
    long double impurity;
    long double gain;
    int64_t separator;
  };
  struct ImpSums {
    long double sumS = 0;
    long double sumS0 = 0;
    long double sumS1 = 0;
    long double sumSqS0 = 0;
    long double sumSqS1 = 0;
  };
  
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height,
                                                  int64_t minLeaf, long double minGain,
                                                  bool useNominalBinary);
  bool isAllSameClass(DataSet& ds);
  AttribResult calcAttribGain(DataSet& ds, int64_t attribInx, long double parentImp,
                              bool useNominalBinary);
  AttribResult calcNominalGain(DataSet& ds, int64_t attribInx, long double parentImp,
                               bool useNominalBinary);
  AttribResult calcNumericGain(DataSet& ds, int64_t attribInx, long double parentImp);
  std::shared_ptr<DecisionTreeNode> createLeaf(DataSet& ds);
  long double calcImpurity(DataSet& ds);
  long double applyFormula(long double sumS, long double sumS0,
                           long double sumS1, long double sumSqS0,
                           long double sumSqS1);
  void calcNormVars(DataSet& ds);
  long double normalizedValue(long double value);

  long double minValue_;
  long double maxValue_;
};
