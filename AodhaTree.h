#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "DecisionTreeNode.h"
#include "Tree.h"

#include <vector>

class ConfigAodha : public ConfigTree {
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
  
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height);
  AttribResult calcAttribGain(DataSet& ds, int64_t attribInx, long double parentImp);
  AttribResult calcNominalGain(DataSet& ds, int64_t attribInx, long double parentImp);
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
