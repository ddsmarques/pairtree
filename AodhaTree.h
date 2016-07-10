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
  void calcNormVars(DataSet& ds);
  long double normalizedValue(long double value);
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height);
  long double calcImpurity(DataSet& ds);
  AttribResult calcAttribGain(DataSet& ds, int64_t attribInx, long double parentImp);
  AttribResult calcNominalGain(DataSet& ds, int64_t attribInx, long double parentImp);
  AttribResult calcNumericGain(DataSet& ds, int64_t attribInx, long double parentImp);
  std::shared_ptr<DecisionTreeNode> createLeaf(DataSet& ds);

  long double minValue_;
  long double maxValue_;
};
